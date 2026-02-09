function simOut = run_stage_manager_test()
%RUN_STAGE_MANAGER_TEST Build a simple harness and simulate StageManager.
% Assumes model 'stage_manager_test' already exists.

model = 'stage_manager_test';
if ~bdIsLoaded(model)
    if exist([model '.slx'], 'file')
        load_system(model);
    else
        new_system(model);
        open_system(model);
        save_system(model);
    end
end
open_system(model);

stageName = 'StageManager';
stagePath = [model '/' stageName];
if isempty(find_system(model, 'SearchDepth', 1, 'Name', stageName))
    create_stage_manager(model);
end

% Clean up old harness blocks if they exist
blocksToDelete = {
    [model '/SM_error'], [model '/SM_upgrade'], [model '/SM_reset'], [model '/SM_enable'], ...
    [model '/SM_stage_offset'], [model '/SM_stage'], [model '/SM_error_sign'], [model '/SM_cooldown'], ...
    [model '/SM_scope']
};
for i = 1:numel(blocksToDelete)
    if getSimulinkBlockHandle(blocksToDelete{i}) ~= -1
        delete_block(blocksToDelete{i});
    end
end

% ===================== Test input design =====================
% Fixed-step dt = 0.05s; total 1.0s.
% 1) reset_flag: 1 at t=0~0.05, then 0; extra reset at t=0.60
% 2) error_value:
%    - 0.05~0.45: +0.5 (steady positive error)
%    - 0.45~0.55: tiny value (deadzone test)
%    - 0.55~0.85: -0.5 (sign flip + negative)
%    - 0.85~1.00: +0.4 (second sign flip)
% 3) should_upgrade pulses at 0.15, 0.20, 0.30, 0.65, 0.70
%    (0.20 should be blocked by cooldown if cooldown=2 frames)
% 4) control_enabled: 1 by default, 0 during 0.75~0.85 (disable test)
t = (0:0.05:1.0)';  % 21 samples

err = zeros(size(t));
err(t >= 0.05 & t < 0.45) = 0.5;
err(t >= 0.45 & t < 0.55) = 1e-7;  % deadzone
err(t >= 0.55 & t < 0.85) = -0.5;
err(t >= 0.85) = 0.4;

upgrade = zeros(size(t));
upgrade(ismember(t, [0.15 0.20 0.30 0.65 0.70])) = 1;

reset_flag = zeros(size(t));
reset_flag(t <= 0.05) = 1;
reset_flag(ismember(t, 0.60)) = 1;

control_enabled = ones(size(t));
control_enabled(t >= 0.75 & t <= 0.85) = 0;

assignin('base', 'sm_error_ts', timeseries(err, t));
assignin('base', 'sm_upgrade_ts', timeseries(upgrade, t));
assignin('base', 'sm_reset_ts', timeseries(reset_flag, t));
assignin('base', 'sm_enable_ts', timeseries(control_enabled, t));

% Sources: From Workspace
add_block('simulink/Sources/From Workspace', [model '/SM_error'], ...
    'Position', [80 80 120 110], 'VariableName', 'sm_error_ts');
add_block('simulink/Sources/From Workspace', [model '/SM_upgrade'], ...
    'Position', [80 150 120 180], 'VariableName', 'sm_upgrade_ts');
add_block('simulink/Sources/From Workspace', [model '/SM_reset'], ...
    'Position', [80 220 120 250], 'VariableName', 'sm_reset_ts');
add_block('simulink/Sources/From Workspace', [model '/SM_enable'], ...
    'Position', [80 290 120 320], 'VariableName', 'sm_enable_ts');

% Sinks: To Workspace
add_block('simulink/Sinks/To Workspace', [model '/SM_stage_offset'], ...
    'Position', [520 80 600 110], 'VariableName', 'sm_stage_offset', 'SaveFormat', 'StructureWithTime');
add_block('simulink/Sinks/To Workspace', [model '/SM_stage'], ...
    'Position', [520 130 600 160], 'VariableName', 'sm_stage', 'SaveFormat', 'StructureWithTime');
add_block('simulink/Sinks/To Workspace', [model '/SM_error_sign'], ...
    'Position', [520 180 600 210], 'VariableName', 'sm_error_sign', 'SaveFormat', 'StructureWithTime');
add_block('simulink/Sinks/To Workspace', [model '/SM_cooldown'], ...
    'Position', [520 230 600 260], 'VariableName', 'sm_cooldown', 'SaveFormat', 'StructureWithTime');

% Optional scope for quick view (4 inputs)
add_block('simulink/Sinks/Scope', [model '/SM_scope'], ...
    'Position', [520 290 600 340]);
set_param([model '/SM_scope'], 'NumInputPorts', '4');
add_line(model, 'StageManager/1', 'SM_scope/1', 'autorouting', 'on');
add_line(model, 'StageManager/2', 'SM_scope/2', 'autorouting', 'on');
add_line(model, 'StageManager/3', 'SM_scope/3', 'autorouting', 'on');
add_line(model, 'StageManager/4', 'SM_scope/4', 'autorouting', 'on');

% Wiring inputs
add_line(model, 'SM_error/1',   'StageManager/1', 'autorouting', 'on');
add_line(model, 'SM_upgrade/1', 'StageManager/2', 'autorouting', 'on');
add_line(model, 'SM_reset/1',   'StageManager/3', 'autorouting', 'on');
add_line(model, 'SM_enable/1',  'StageManager/4', 'autorouting', 'on');

% Wiring outputs to workspace
add_line(model, 'StageManager/1', 'SM_stage_offset/1', 'autorouting', 'on');
add_line(model, 'StageManager/2', 'SM_stage/1', 'autorouting', 'on');
add_line(model, 'StageManager/3', 'SM_error_sign/1', 'autorouting', 'on');
add_line(model, 'StageManager/4', 'SM_cooldown/1', 'autorouting', 'on');

% Solver settings
set_param(model, 'Solver', 'FixedStepDiscrete', 'FixedStep', '0.05', 'StopTime', '1');

% Run simulation
simOut = sim(model);
assignin('base', 'sm_simout', simOut);

fprintf('StageManager test complete. Workspace vars: sm_stage_offset, sm_stage, sm_error_sign, sm_cooldown\n');
end
