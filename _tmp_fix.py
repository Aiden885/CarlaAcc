from pathlib import Path
path = Path('generate_output_formatter_code.m')
lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
for i, line in enumerate(lines):
    if '构造' in line:
        lines[i] = '%% Build integrated output struct'
    elif '按照总线' in line:
        lines[i] = '%% Assign fields following bus ordering (codegen friendly)'
    elif 'DecisionSPPVTOutputExtended' in line and '总线' in line:
        lines[i] = '% Follow DecisionSPPVTOutputExtended bus ordering (18 fields)'
    elif '1-6' in line and '信' in line:
        lines[i] = '% 1-6: Decision module signals'
    elif '7-11' in line:
        lines[i] = '% 7-11: SPPVT controller outputs (double precision)'
    elif '注意' in line:
        lines[i] = '% Note: sppvt_output4 is jerk (not stage); stage comes from Stage_Manager'
    elif '12:' in line:
        lines[i] = '% 12: Debug information (composite code)'
    elif '13-15' in line:
        lines[i] = '% 13-15: Decision module state outputs'
    elif '16-18' in line:
        lines[i] = '% 16-18: SPPVT state outputs consumed in Python'
    elif 'Stage offset' in line:
        lines[i] = '% 16: Stage offset (scalar)'
    elif 'Stage Manager' in line and 'new_stage' in line:
        lines[i] = '% 17: Stage Manager state [new_stage, new_error_sign, new_upgrade_count]'
    elif 'Adapter' in line and '数组' in line:
        lines[i] = '% 18: Adapter state [new_control_error, new_velocity, new_accel]'
    elif '修正' in line:
        lines[i] = '% Correction: replace placeholder velocity/accel with SPPVT derivatives'
    elif '调试：' in line or '调试:' in line:
        lines[i] = '% Debug: dump adapter_states_out shape & values'
    elif '误差的一阶' in line:
        lines[i] = 'adapter_states_corrected(2) = double(sppvt_output2);  % new_velocity = first derivative of error'
    elif '误差的二阶' in line:
        lines[i] = 'adapter_states_corrected(3) = double(sppvt_output3);  % new_accel = second derivative of error'
    elif "fprintf('\\]\\n');" in line:
        lines[i] = "fprintf(']\\n');"
    elif '维度检查' in line:
        lines[i] = '% Dimension check - ensure 3 elements'
    elif '调试输出' in line and '18' in line:
        lines[i] = '% Debug output (18-field externalization summary)'
    elif '决策状态' in line:
        lines[i] = '% Decision state debug output'
    elif 'SPPVT' in line and '状态外化' in line:
        lines[i] = '% SPPVT state debug output'

text = '\n'.join(lines) + '\n'
text = text.replace("fprintf(']\\n');", "fprintf(']\n');")
text = text.replace("fprintf(']\n');", "fprintf(']\n');")
text = text.replace("fprintf(']\n');", "fprintf(']\n');")
text = text.replace("fprintf(']\n');", "fprintf(']\n');")
text = text.replace("fprintf(']\n');", "fprintf(']\n');")
text = text.replace("fprintf(']\n');", "fprintf(']\n');")
text = text.replace("fprintf(']\n');", "fprintf(']\n');")
text = text.replace("fprintf(']\n');", "fprintf(']\n');")
text = text.replace("fprintf(']\n');", "fprintf(']\n');")
path.write_text(text.replace("fprintf(']\n');", "fprintf(']\n');").replace("fprintf(']\n');", "fprintf(']\n');"), encoding='utf-8')
