function report = check_integrated_model_wiring(model_name)
%CHECK_INTEGRATED_MODEL_WIRING Validate acc_integrated_model top-level wiring
% Usage:
%   report = check_integrated_model_wiring();
%   report = check_integrated_model_wiring('acc_integrated_model');
%
% This script checks:
%   1) UDP Receive/Send data sizes (7 in, 6 out)
%   2) Demux output order from UDP Receive
%   3) Decision block input count (expect 4)
%   4) Mux order into UDP Send (expect 6)
%   5) Basic wiring consistency (no obvious swaps)

    if nargin < 1 || isempty(model_name)
        model_name = 'acc_integrated_model';
    end

    report = struct();
    report.warnings = {};
    report.info = {};

    fprintf('=== Wiring Check: %s ===\n', model_name);

    % Load model (do not open UI)
    load_system(model_name);

    % ---------- Find UDP Receive ----------
    udp_recv = find_system(model_name, 'MaskType', 'UDP Receive');
    if isempty(udp_recv)
        udp_recv = find_system(model_name, 'Name', 'UDP Receive');
    end
    if isempty(udp_recv)
        report.warnings{end+1} = 'UDP Receive block not found.';
        disp(report.warnings{end});
        return;
    end
    udp_recv = udp_recv{1};

    % ---------- Find UDP Send ----------
    udp_send = find_system(model_name, 'MaskType', 'UDP Send');
    if isempty(udp_send)
        udp_send = find_system(model_name, 'Name', 'UDP Send');
    end
    if isempty(udp_send)
        report.warnings{end+1} = 'UDP Send block not found.';
        disp(report.warnings{end});
        return;
    end
    udp_send = udp_send{1};

    % ---------- Check UDP sizes ----------
    try
        recv_size = get_param(udp_recv, 'DataSize');
        fprintf('UDP Receive DataSize: %s\n', recv_size);
        recv_n = parse_first_int(recv_size);
        if isempty(recv_n) || recv_n ~= 7
            report.warnings{end+1} = sprintf('UDP Receive DataSize expected 7, got %s.', recv_size);
        end
    catch
        report.warnings{end+1} = 'Failed to read UDP Receive DataSize.';
    end

    try
        send_size = get_param(udp_send, 'DataSize');
        fprintf('UDP Send DataSize: %s\n', send_size);
        send_n = parse_first_int(send_size);
        if isempty(send_n) || send_n ~= 6
            report.warnings{end+1} = sprintf('UDP Send DataSize expected 6, got %s.', send_size);
        end
    catch
        report.warnings{end+1} = 'Failed to read UDP Send DataSize.';
    end

    % ---------- Find Demux connected to UDP Receive ----------
    demux = find_demux_connected(udp_recv);
    if isempty(demux)
        report.warnings{end+1} = 'Demux connected to UDP Receive not found.';
    else
        fprintf('Demux: %s\n', demux);
    end

    % ---------- Find Mux connected to UDP Send ----------
    mux = find_mux_connected(udp_send);
    if isempty(mux)
        report.warnings{end+1} = 'Mux connected to UDP Send not found.';
    else
        fprintf('Mux: %s\n', mux);
    end

    % ---------- Identify Decision subsystem ----------
    decision_blk = find_subsystem_by_name(model_name, 'Decision');
    decision_inport_names = {};
    if isempty(decision_blk)
        report.warnings{end+1} = 'Decision subsystem not found (top-level name "Decision").';
    else
        fprintf('Decision subsystem: %s\n', decision_blk);
        [decision_inport_names, decision_inport_ports] = list_subsystem_inports(decision_blk);
        fprintf('Decision inports (%d):\n', numel(decision_inport_names));
        for i = 1:numel(decision_inport_names)
            fprintf('  In%d: %s\n', decision_inport_ports(i), decision_inport_names{i});
        end
        if numel(decision_inport_names) ~= 4
            report.warnings{end+1} = sprintf('Decision inports expected 4, got %d.', numel(decision_inport_names));
        end
    end

    % ---------- Demux mapping check ----------
    if ~isempty(demux)
        expected_names = {'current_state','command_type','has_history','last_active_decision', ...
                          'control_error','Y0','reset_flag'};
        fprintf('\nDemux output mapping:\n');
        demux_ports = get_param(demux, 'PortHandles');
        outports = demux_ports.Outport;
        for i = 1:numel(outports)
            [dst_names, dst_ports] = describe_destinations(outports(i));
            label = expected_names{min(i, numel(expected_names))};
            fprintf('  Out%d (%s) -> %s\n', i, label, join_dest(dst_names, dst_ports));

            % Heuristic warnings
            if i == 2 && ~any(contains(lower(dst_names), 'decision'))
                report.warnings{end+1} = 'Demux Out2 (command_type) not routed to Decision block.';
            end
            if i == 7 && any(contains(lower(dst_names), 'decision'))
                report.warnings{end+1} = 'Demux Out7 (reset_flag) routed into Decision block (should not).';
            end

            % If routed into Decision, verify inport name matches expected
            for d = 1:numel(dst_names)
                if ~isempty(decision_blk) && startsWith(dst_names{d}, decision_blk)
                    if d <= numel(dst_ports)
                        port_num = str2double(dst_ports{d});
                    else
                        port_num = NaN;
                    end
                    in_name = lookup_inport_name(decision_inport_names, decision_inport_ports, port_num);
                    if ~isempty(in_name)
                        if ~contains(lower(in_name), label)
                            report.warnings{end+1} = sprintf( ...
                                'Decision inport mismatch: Demux Out%d (%s) -> Decision In%d (%s).', ...
                                i, label, port_num, in_name);
                        end
                    end
                end
            end
        end
    end

    % ---------- Decision internal checks ----------
    if ~isempty(decision_blk)
        fprintf('\nDecision internal wiring:\n');
        % Warn if any stateful blocks exist
        stateful = find_system(decision_blk, 'RegExp', 'on', ...
            'BlockType', 'UnitDelay|Memory|Delay|DiscreteTransferFcn');
        if ~isempty(stateful)
            report.warnings{end+1} = 'Decision subsystem contains stateful blocks (UnitDelay/Memory/Delay).';
            for i = 1:numel(stateful)
                fprintf('  ⚠️ stateful: %s\n', stateful{i});
            end
        end

        % Locate LUT blocks and trace their inputs
        luts = find_system(decision_blk, 'RegExp', 'on', 'BlockType', 'Lookup.*');
        if isempty(luts)
            % fallback: blocks with name containing LUT_
            luts = find_system(decision_blk, 'RegExp', 'on', 'Name', 'LUT_.*');
        end
        for i = 1:numel(luts)
            blk = luts{i};
            ph = get_param(blk, 'PortHandles');
            if isempty(ph.Inport)
                continue;
            end
            fprintf('  LUT: %s\n', blk);
            for p = 1:numel(ph.Inport)
                [src_name, src_port] = trace_source(ph.Inport(p), decision_blk, 6);
                fprintf('    In%d <- %s\n', p, join_src(src_name, src_port));
                if p == 1 && ~contains(lower(src_name), 'current_state')
                    report.warnings{end+1} = sprintf('LUT input1 not driven by current_state: %s', blk);
                end
                if p == 2 && ~contains(lower(src_name), 'command_type')
                    report.warnings{end+1} = sprintf('LUT input2 not driven by command_type: %s', blk);
                end
            end
        end
    end

    % ---------- Mux mapping check ----------
    if ~isempty(mux)
        fprintf('\nMux input mapping (to UDP Send):\n');
        mux_ports = get_param(mux, 'PortHandles');
        inports = mux_ports.Inport;
        for i = 1:numel(inports)
            [src_name, src_port] = describe_source(inports(i));
            fprintf('  In%d <- %s\n', i, join_src(src_name, src_port));
            if i <= 5 && ~contains(lower(src_name), 'decision')
                report.warnings{end+1} = sprintf('Mux In%d expected Decision output, got %s.', i, src_name);
            end
            if i == 6 && contains(lower(src_name), 'decision')
                report.warnings{end+1} = 'Mux In6 should be SPPVT control_output, but comes from Decision.';
            end
        end
    end

    % ---------- Report warnings ----------
    if isempty(report.warnings)
        fprintf('\nNo obvious wiring issues detected.\n');
    else
        fprintf('\nWarnings:\n');
        for i = 1:numel(report.warnings)
            fprintf('  - %s\n', report.warnings{i});
        end
    end

    fprintf('=== Done ===\n');
end

% ========================= Helpers =========================

function demux = find_demux_connected(src_block)
    demux = '';
    try
        ph = get_param(src_block, 'PortHandles');
        out = ph.Outport;
        if isempty(out)
            return;
        end
        line = get_param(out(1), 'Line');
        if line == -1
            return;
        end
        dst = get_param(line, 'DstBlockHandle');
        if isempty(dst)
            return;
        end
        for k = 1:numel(dst)
            if strcmp(get_param(dst(k), 'BlockType'), 'Demux')
                demux = getfullname(dst(k));
                return;
            end
        end
    catch
        demux = '';
    end
end

function mux = find_mux_connected(dst_block)
    mux = '';
    try
        ph = get_param(dst_block, 'PortHandles');
        in = ph.Inport;
        if isempty(in)
            return;
        end
        line = get_param(in(1), 'Line');
        if line == -1
            return;
        end
        src = get_param(line, 'SrcBlockHandle');
        if isempty(src)
            return;
        end
        if strcmp(get_param(src, 'BlockType'), 'Mux')
            mux = getfullname(src);
        end
    catch
        mux = '';
    end
end

function blk = find_subsystem_by_name(model_name, name_exact)
    blk = '';
    blocks = find_system(model_name, 'SearchDepth', 1, 'BlockType', 'SubSystem', 'Name', name_exact);
    if isempty(blocks)
        return;
    end
    blk = blocks{1};
end

function [names, ports] = list_subsystem_inports(subsys_path)
    names = {};
    ports = [];
    try
        inports = find_system(subsys_path, 'SearchDepth', 1, 'BlockType', 'Inport');
        if isempty(inports)
            return;
        end
        tmp = cell(numel(inports), 2);
        for i = 1:numel(inports)
            pnum = str2double(get_param(inports{i}, 'Port'));
            tmp{i,1} = pnum;
            tmp{i,2} = get_param(inports{i}, 'Name');
        end
        % sort by port number
        tmp = sortrows(tmp, 1);
        ports = cell2mat(tmp(:,1));
        names = tmp(:,2);
    catch
        names = {};
        ports = [];
    end
end

function n = count_inports(block_path)
    n = 0;
    try
        ph = get_param(block_path, 'PortHandles');
        n = numel(ph.Inport);
    catch
        n = 0;
    end
end

function name = lookup_inport_name(names, ports, port_num)
    name = '';
    if isempty(names) || isempty(ports) || isnan(port_num)
        return;
    end
    idx = find(ports == port_num, 1, 'first');
    if ~isempty(idx)
        name = names{idx};
    end
end

function n = parse_first_int(s)
    n = [];
    try
        if isnumeric(s)
            n = s(1);
            return;
        end
        t = regexp(char(s), '(-?\\d+)', 'tokens', 'once');
        if ~isempty(t)
            n = str2double(t{1});
        end
    catch
        n = [];
    end
end

function [dst_names, dst_ports] = describe_destinations(outport_handle)
    dst_names = {};
    dst_ports = {};
    try
        line = get_param(outport_handle, 'Line');
        if line == -1
            return;
        end
        dst = get_param(line, 'DstBlockHandle');
        dstp = get_param(line, 'DstPortHandle');
        if isempty(dst)
            return;
        end
        for i = 1:numel(dst)
            dst_names{end+1} = getfullname(dst(i));
            if numel(dstp) >= i
                dst_ports{end+1} = num2str(get_param(dstp(i), 'PortNumber'));
            else
                dst_ports{end+1} = '?';
            end
        end
    catch
        % ignore
    end
end

function [src_name, src_port] = describe_source(inport_handle)
    src_name = '';
    src_port = '';
    try
        line = get_param(inport_handle, 'Line');
        if line == -1
            return;
        end
        src = get_param(line, 'SrcBlockHandle');
        srcp = get_param(line, 'SrcPortHandle');
        if src ~= -1
            src_name = getfullname(src);
        end
        if srcp ~= -1
            src_port = num2str(get_param(srcp, 'PortNumber'));
        end
    catch
        % ignore
    end
end

function [src_name, src_port] = trace_source(inport_handle, scope, max_hops)
    % Trace line source upstream until an Inport or max_hops
    src_name = '';
    src_port = '';
    if max_hops <= 0
        return;
    end
    try
        line = get_param(inport_handle, 'Line');
        if line == -1
            return;
        end
        src = get_param(line, 'SrcBlockHandle');
        srcp = get_param(line, 'SrcPortHandle');
        if src == -1
            return;
        end
        src_name = getfullname(src);
        if srcp ~= -1
            src_port = num2str(get_param(srcp, 'PortNumber'));
        end

        % If source is Data Type Conversion or similar, keep tracing
        blk_type = get_param(src, 'BlockType');
        if strcmp(blk_type, 'Inport')
            return;
        end
        if any(strcmp(blk_type, {'DataTypeConversion','Gain','Saturate','Math','Abs','Switch'}))
            ph = get_param(src, 'PortHandles');
            if ~isempty(ph.Inport)
                [src_name, src_port] = trace_source(ph.Inport(1), scope, max_hops-1);
            end
        end
    catch
        % ignore
    end
end

function s = join_dest(names, ports)
    if isempty(names)
        s = '<unconnected>';
        return;
    end
    parts = cell(1, numel(names));
    for i = 1:numel(names)
        p = ports{i};
        parts{i} = sprintf('%s:port%s', names{i}, p);
    end
    s = strjoin(parts, ' | ');
end

function s = join_src(name, port)
    if isempty(name)
        s = '<unconnected>';
    else
        if isempty(port)
            s = name;
        else
            s = sprintf('%s:port%s', name, port);
        end
    end
end
