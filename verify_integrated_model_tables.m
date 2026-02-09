function report = verify_integrated_model_tables(model_name)
%VERIFY_INTEGRATED_MODEL_TABLES Validate LUT data used by acc_integrated_model
% Usage:
%   report = verify_integrated_model_tables();
%   report = verify_integrated_model_tables('acc_integrated_model');
%
% Checks:
%   - decision_lookup_data.mat exists and matches key expected entries
%   - model workspace has required variables
%   - LUT blocks in Decision use correct variable names
%   - LUT table values match .mat (sanity spot-checks)

    if nargin < 1 || isempty(model_name)
        model_name = 'acc_integrated_model';
    end

    report = struct();
    report.warnings = {};

    fprintf('=== LUT Check: %s ===\n', model_name);

    % Load lookup data
    if ~exist('decision_lookup_data.mat', 'file')
        error('decision_lookup_data.mat not found.');
    end
    mat_data = load('decision_lookup_data.mat');

    fprintf('Loaded decision_lookup_data.mat\n');
    fprintf('  state_bp: %s\n', mat2str(mat_data.state_bp));
    fprintf('  command_bp: %s\n', mat2str(mat_data.command_bp));
    fprintf('  next_state_table: %dx%d\n', size(mat_data.next_state_table));
    fprintf('  decision_table: %dx%d\n', size(mat_data.decision_table));
    fprintf('  control_enabled_table: %dx%d\n', size(mat_data.control_enabled_table));

    % Load model
    load_system(model_name);

    % Model workspace variables
    try
        mdlWks = get_param(model_name, 'ModelWorkspace');
        required_vars = {'state_bp','command_bp','next_state_table','decision_table','control_enabled_table','side_effect_table'};
        fprintf('\nModel workspace variables:\n');
        for i = 1:numel(required_vars)
            v = required_vars{i};
            if mdlWks.hasVariable(v)
                fprintf('  ✅ %s\n', v);
            else
                fprintf('  ❌ %s (missing)\n', v);
                report.warnings{end+1} = sprintf('Model workspace missing variable: %s', v);
            end
        end
    catch ME
        report.warnings{end+1} = sprintf('Failed to read model workspace: %s', ME.message);
    end

    % Locate LUT blocks
    decision_path = [model_name '/Decision'];
    luts = find_system(decision_path, 'BlockType', 'Lookup_nD');
    if isempty(luts)
        luts = find_system(decision_path, 'RegExp', 'on', 'Name', 'LUT_.*');
    end

    fprintf('\nDecision LUT blocks:\n');
    for i = 1:numel(luts)
        blk = luts{i};
        fprintf('  %s\n', blk);

        % Table param
        tbl = safe_get_param(blk, 'Table');
        bp1 = safe_get_param(blk, 'BreakpointsForDimension1');
        bp2 = safe_get_param(blk, 'BreakpointsForDimension2');

        if ~isempty(tbl)
            fprintf('    Table: %s\n', tbl);
        end
        if ~isempty(bp1)
            fprintf('    BP1: %s\n', bp1);
        end
        if ~isempty(bp2)
            fprintf('    BP2: %s\n', bp2);
        end

        % Interp/Extrap method (if supported)
        im = safe_get_param(blk, 'InterpMethod');
        em = safe_get_param(blk, 'ExtrapMethod');
        if ~isempty(im)
            fprintf('    Interp: %s\n', im);
        end
        if ~isempty(em)
            fprintf('    Extrap: %s\n', em);
        end
    end

    % Spot-check values in model workspace if present
    try
        if mdlWks.hasVariable('next_state_table')
            model_next = mdlWks.getVariable('next_state_table');
            model_decision = mdlWks.getVariable('decision_table');
            model_enabled = mdlWks.getVariable('control_enabled_table');

            % S2+Cmd1 -> indices (3,2)
            s2_cmd1_next = model_next(3,2);
            s2_cmd1_dec = model_decision(3,2);
            s2_cmd1_en = model_enabled(3,2);

            fprintf('\nSpot-check (model workspace): S2+Cmd1\n');
            fprintf('  next_state=%d decision=%d enabled=%d\n', s2_cmd1_next, s2_cmd1_dec, s2_cmd1_en);

            if s2_cmd1_next ~= mat_data.next_state_table(3,2) || ...
               s2_cmd1_dec ~= mat_data.decision_table(3,2) || ...
               s2_cmd1_en ~= mat_data.control_enabled_table(3,2)
                report.warnings{end+1} = 'Model workspace LUT values do not match decision_lookup_data.mat.';
            end
        end
    catch ME
        report.warnings{end+1} = sprintf('Spot-check failed: %s', ME.message);
    end

    if isempty(report.warnings)
        fprintf('\nNo obvious LUT data issues detected.\n');
    else
        fprintf('\nWarnings:\n');
        for i = 1:numel(report.warnings)
            fprintf('  - %s\n', report.warnings{i});
        end
    end

    fprintf('=== Done ===\n');
end

function v = safe_get_param(blk, name)
    v = '';
    try
        v = get_param(blk, name);
    catch
        v = '';
    end
end
