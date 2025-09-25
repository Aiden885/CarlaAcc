function diagnose_bus_issues()
%DIAGNOSE_BUS_ISSUES 诊断并修复14/15-field总线集成问题
% 检查当前工作空间中的总线定义，并诊断Simulink模型的总线引用

fprintf('=== 14/15-Field总线集成诊断工具 ===\n\n');

%% 1. 检查当前工作空间中的总线定义
fprintf('1. 检查工作空间中的总线定义...\n');

% 检查旧总线是否存在
old_buses = {'DecisionSPPVTInput', 'DecisionSPPVTOutput'};
new_buses = {'DecisionSPPVTInputExtended', 'DecisionSPPVTOutputExtended'};

fprintf('   旧总线状态:\n');
for i = 1:length(old_buses)
    bus_name = old_buses{i};
    if evalin('base', sprintf('exist(''%s'', ''var'')', bus_name))
        fprintf('   ❌ %s: 存在 (应该删除)\n', bus_name);
    else
        fprintf('   ✅ %s: 不存在 (正确)\n', bus_name);
    end
end

fprintf('   新总线状态:\n');
for i = 1:length(new_buses)
    bus_name = new_buses{i};
    if evalin('base', sprintf('exist(''%s'', ''var'')', bus_name))
        fprintf('   ✅ %s: 存在 (正确)\n', bus_name);
    else
        fprintf('   ❌ %s: 不存在 (需要创建)\n', bus_name);
    end
end

%% 2. 创建14/15-field总线定义
fprintf('\n2. 创建/更新14/15-field总线定义...\n');
try
    create_decision_sppvt_bus();
    fprintf('   ✅ 总线定义创建成功\n');

    % 验证总线字段数量
    if evalin('base', 'exist(''DecisionSPPVTInputExtended'', ''var'')')
        input_bus = evalin('base', 'DecisionSPPVTInputExtended');
        fprintf('   📊 DecisionSPPVTInputExtended: %d个字段\n', length(input_bus.Elements));

        if length(input_bus.Elements) == 14
            fprintf('   ✅ 输入总线字段数量正确 (14个)\n');
        else
            fprintf('   ❌ 输入总线字段数量不正确 (期望14个，实际%d个)\n', length(input_bus.Elements));
        end
    end

    if evalin('base', 'exist(''DecisionSPPVTOutputExtended'', ''var'')')
        output_bus = evalin('base', 'DecisionSPPVTOutputExtended');
        fprintf('   📊 DecisionSPPVTOutputExtended: %d个字段\n', length(output_bus.Elements));

        if length(output_bus.Elements) == 15
            fprintf('   ✅ 输出总线字段数量正确 (15个)\n');
        else
            fprintf('   ❌ 输出总线字段数量不正确 (期望15个，实际%d个)\n', length(output_bus.Elements));
        end
    end

catch ME
    fprintf('   ❌ 总线创建失败: %s\n', ME.message);
    return;
end

%% 3. 检查Simulink模型
model_name = 'ACC_Decision_SPPVT_Integrated';
fprintf('\n3. 检查Simulink模型: %s\n', model_name);

% 检查模型文件是否存在
if exist([model_name '.slx'], 'file')
    fprintf('   ✅ 模型文件存在\n');
else
    fprintf('   ❌ 模型文件不存在: %s.slx\n', model_name);
    fprintf('   💡 建议: 请确认模型文件路径是否正确\n');
    return;
end

% 尝试加载模型
try
    fprintf('   🔄 尝试加载模型...\n');
    load_system(model_name);
    fprintf('   ✅ 模型加载成功\n');

    % 检查模型中的Inport和Outport配置
    fprintf('   🔍 检查模型端口配置...\n');

    % 查找所有Inport块
    inports = find_system(model_name, 'BlockType', 'Inport');
    fprintf('   📥 发现 %d 个输入端口:\n', length(inports));

    for i = 1:length(inports)
        port_name = get_param(inports{i}, 'Name');
        try
            bus_name = get_param(inports{i}, 'BusOutputAsStruct');
            if strcmp(bus_name, 'on')
                bus_object = get_param(inports{i}, 'BusObject');
                fprintf('      %s: 使用总线 "%s"\n', port_name, bus_object);

                if strcmp(bus_object, 'DecisionSPPVTInputExtended')
                    fprintf('         ✅ 使用正确的输入总线\n');
                elseif strcmp(bus_object, 'DecisionSPPVTInput')
                    fprintf('         ❌ 使用旧的输入总线，需要更新\n');
                else
                    fprintf('         ⚠️  使用未知总线: %s\n', bus_object);
                end
            else
                fprintf('      %s: 非总线端口\n', port_name);
            end
        catch
            fprintf('      %s: 无法获取总线信息\n', port_name);
        end
    end

    % 查找所有Outport块
    outports = find_system(model_name, 'BlockType', 'Outport');
    fprintf('   📤 发现 %d 个输出端口:\n', length(outports));

    for i = 1:length(outports)
        port_name = get_param(outports{i}, 'Name');
        try
            bus_name = get_param(outports{i}, 'BusOutputAsStruct');
            if strcmp(bus_name, 'on')
                bus_object = get_param(outports{i}, 'BusObject');
                fprintf('      %s: 使用总线 "%s"\n', port_name, bus_object);

                if strcmp(bus_object, 'DecisionSPPVTOutputExtended')
                    fprintf('         ✅ 使用正确的输出总线\n');
                elseif strcmp(bus_object, 'DecisionSPPVTOutput')
                    fprintf('         ❌ 使用旧的输出总线，需要更新\n');
                else
                    fprintf('         ⚠️  使用未知总线: %s\n', bus_object);
                end
            else
                fprintf('      %s: 非总线端口\n', port_name);
            end
        catch
            fprintf('      %s: 无法获取总线信息\n', port_name);
        end
    end

    close_system(model_name, 0);

catch ME
    fprintf('   ❌ 模型检查失败: %s\n', ME.message);

    if contains(ME.message, 'Bus object')
        fprintf('\n💡 诊断建议:\n');
        fprintf('   1. 模型可能引用了不存在的总线对象\n');
        fprintf('   2. 需要在Simulink中手动更新Inport/Outport的总线配置\n');
        fprintf('   3. 将旧的总线名称替换为新的14/15-field总线名称:\n');
        fprintf('      - DecisionSPPVTInput → DecisionSPPVTInputExtended\n');
        fprintf('      - DecisionSPPVTOutput → DecisionSPPVTOutputExtended\n');

        fprintf('\n🔧 修复步骤:\n');
        fprintf('   1. 在Simulink中打开模型: %s.slx\n', model_name);
        fprintf('   2. 双击每个Inport块，将Bus object改为 DecisionSPPVTInputExtended\n');
        fprintf('   3. 双击每个Outport块，将Bus object改为 DecisionSPPVTOutputExtended\n');
        fprintf('   4. 保存模型并重新测试\n');
    end
end

%% 4. 提供解决方案
fprintf('\n=== 解决方案总结 ===\n');
fprintf('✅ 总线定义正常 - 14/15-field总线已创建\n');

if exist([model_name '.slx'], 'file')
    fprintf('⚠️  Simulink模型配置 - 可能需要手动更新端口总线配置\n');
    fprintf('💡 请在Simulink中检查并更新Inport/Outport的总线对象引用\n');
else
    fprintf('❌ Simulink模型文件缺失 - 需要确认文件路径\n');
end

fprintf('\n测试建议:\n');
fprintf('1. 运行此脚本确保所有总线定义正确\n');
fprintf('2. 在Simulink中手动验证模型端口配置\n');
fprintf('3. 再次运行Python测试脚本\n');

end