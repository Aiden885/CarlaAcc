  function [new_stage_offset, new_stage, sign_changed, stage_manager_states_out] = fcn(should_upgrade, error_value, sppvt_rho, prev_stage_offset, stage_manager_states_in)
  %#codegen
  % Stateless SPPVT Stage Manager with array-based state management
  % Compatible with 17/18-field bus design (3-element state array version)

  % Extract states from input array (replaces persistent variables)
  current_stage = int32(stage_manager_states_in(1));
  prev_error_sign = int32(stage_manager_states_in(2));
  upgrade_count = int32(stage_manager_states_in(3));

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

  % Output state array for next cycle (to be captured by Python)
  % 3-element state array: [new_stage, new_error_sign, upgrade_count]
  stage_manager_states_out = [double(new_stage); double(new_error_sign); double(upgrade_count)];

  end
