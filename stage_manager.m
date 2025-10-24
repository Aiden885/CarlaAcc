  function [new_stage_offset, new_stage, sign_changed, new_error_sign_out, new_upgrade_count_out] = fcn(should_upgrade, error_value, sppvt_rho, prev_stage_offset, external_stage, external_error_sign, external_upgrade_count, com1, com2, com3)
  %#codegen
  % Stateless SPPVT Stage Manager with scalar-based state management
  % Compatible with 21/22-field bus design (all scalar version)

  % Extract states from input scalars (replaces persistent variables and array extraction)
  current_stage = int32(external_stage);
  prev_error_sign = int32(external_error_sign);
  upgrade_count = int32(external_upgrade_count);

  current_stage_offset = prev_stage_offset;

  % Calculate current error sign
  if abs(error_value) < 1e-6
      current_sign = int32(0);     % Near zero
  elseif error_value > 0
      current_sign = int32(1);     % Positive error
  else
      current_sign = int32(-1);    % Negative error
  end

  % Initialize outputs with current values
  new_stage = current_stage;
  sign_changed = false;

  % Error sign change detection
  if (prev_error_sign ~= 0) && (current_sign ~= 0) && (prev_error_sign ~= current_sign)
      % Sign change: reset to initial stage
      sign_changed = true;

      % Reset all states to initial level
      new_stage = int32(1);
      new_stage_offset = 0.0;      % Initial stage offset is 0
      upgrade_count = int32(0);

  elseif should_upgrade && ~sign_changed
      % No sign change and upgrade condition met
      new_stage = new_stage + int32(1);
      upgrade_count = upgrade_count + int32(1);

      % Calculate new stage offset based on error sign (consistent with existing logic)
      if error_value > 0
          % Positive error: increase positive stage offset
          new_stage_offset = prev_stage_offset + sppvt_rho * abs(error_value);
      else
          % Negative error: increase negative stage offset
          new_stage_offset = prev_stage_offset - sppvt_rho * abs(error_value);
      end

      % Limit stage offset range [-100, 100] to prevent infinite accumulation
      new_stage_offset = max(-100.0, min(100.0, new_stage_offset));

  else
      % No upgrade: maintain current state
      new_stage_offset = prev_stage_offset;
  end

  % Update error sign history (only for non-zero errors)
  new_error_sign = prev_error_sign;
  if current_sign ~= 0
      new_error_sign = current_sign;
  end

  % Output individual scalars for next cycle (to be captured by Python)
  % Scalar outputs: new_error_sign_out, new_upgrade_count_out (new_stage already in output list)
  new_error_sign_out = double(new_error_sign);
  new_upgrade_count_out = double(upgrade_count);

  % Print the three conditional variables using fprintf
  % Print the three conditional variables as 'true' or 'false' with detailed labels
  fprintf('Condition 1 (com1): %s\n', string(com1));
  fprintf('Condition 2 (com2): %f\n', double(com2));
  fprintf('Condition 3 (com3): %s\n', string(com3));
  end
