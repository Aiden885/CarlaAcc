function test_model_creation()
%% 测试ACC Stateflow模型创建的基本功能
% 逐步验证模型创建过程

    fprintf('=== ACC Stateflow模型创建测试 ===\n');
    
    % 测试1: 基本模型创建
    fprintf('\n1. 测试基本模型创建...\n');
    test_basic_model_creation();
    
    % 测试2: Chart数据配置
    fprintf('\n2. 测试Chart数据配置...\n');
    test_chart_data_configuration();
    
    % 测试3: 手动连接测试
    fprintf('\n3. 测试手动连接...\n');
    test_manual_connections();
    
    fprintf('\n=== 测试完成 ===\n');
end

function test_basic_model_creation()
    model_name = 'acc_test_model';
    
    % 清理
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
    if exist([model_name '.slx'], 'file')
        delete([model_name '.slx']);
    end
    
    try
        % 创建模型
        new_system(model_name);
        fprintf('  ✅ 模型创建成功\n');
        
        % 添加Chart
        chart_block = add_block('sflib/Chart', [model_name '/Test_Chart']);
        set_param(chart_block, 'Position', [100 100 200 200]);
        fprintf('  ✅ Stateflow Chart添加成功\n');
        
        % 添加输入输出端口
        in1 = add_block('simulink/Sources/In1', [model_name '/Input1']);
        set_param(in1, 'Position', [50 150 80 170]);
        
        out1 = add_block('simulink/Sinks/Out1', [model_name '/Output1']);
        set_param(out1, 'Position', [250 150 280 170]);
        
        fprintf('  ✅ 输入输出端口添加成功\n');
        
        % 保存模型
        save_system(model_name);
        open_system(model_name);
        
        fprintf('  ✅ 模型保存和打开成功\n');
        
        % 清理
        close_system(model_name, 0);
        delete([model_name '.slx']);
        
    catch ME
        fprintf('  ❌ 基本模型创建失败: %s\n', ME.message);
        if bdIsLoaded(model_name)
            close_system(model_name, 0);
        end
    end
end

function test_chart_data_configuration()
    model_name = 'acc_chart_test';
    
    % 清理
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
    if exist([model_name '.slx'], 'file')
        delete([model_name '.slx']);
    end
    
    try
        % 创建模型和Chart
        new_system(model_name);
        chart_block = add_block('sflib/Chart', [model_name '/Test_Chart']);
        
        % 获取Chart对象
        rt = sfroot;
        chart = rt.find('-isa', 'Stateflow.Chart', 'Path', [model_name '/Test_Chart']);
        
        if isempty(chart)
            error('无法找到Chart对象');
        end
        
        % 添加测试数据
        test_input = Stateflow.Data(chart);
        test_input.Name = 'test_input';
        test_input.Scope = 'Input';
        test_input.DataType = 'double';
        test_input.Props.InitialValue = '0';
        
        test_output = Stateflow.Data(chart);
        test_output.Name = 'test_output';
        test_output.Scope = 'Output';
        test_output.DataType = 'double';
        test_output.Props.InitialValue = '0';
        
        fprintf('  ✅ Chart数据配置成功\n');
        
        % 保存
        save_system(model_name);
        
        % 清理
        close_system(model_name, 0);
        delete([model_name '.slx']);
        
    catch ME
        fprintf('  ❌ Chart数据配置失败: %s\n', ME.message);
        if bdIsLoaded(model_name)
            close_system(model_name, 0);
        end
    end
end

function test_manual_connections()
    model_name = 'acc_connection_test';
    
    % 清理
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
    if exist([model_name '.slx'], 'file')
        delete([model_name '.slx']);
    end
    
    try
        % 创建模型
        new_system(model_name);
        
        % 添加组件
        in1 = add_block('simulink/Sources/In1', [model_name '/Input1']);
        set_param(in1, 'Position', [50 150 80 170]);
        
        chart_block = add_block('sflib/Chart', [model_name '/Test_Chart']);
        set_param(chart_block, 'Position', [150 100 250 200]);
        
        out1 = add_block('simulink/Sinks/Out1', [model_name '/Output1']);
        set_param(out1, 'Position', [300 150 330 170]);
        
        % 配置Chart数据
        rt = sfroot;
        chart = rt.find('-isa', 'Stateflow.Chart', 'Path', [model_name '/Test_Chart']);
        
        if ~isempty(chart)
            % 添加输入输出数据
            input_data = Stateflow.Data(chart);
            input_data.Name = 'u';
            input_data.Scope = 'Input';
            input_data.DataType = 'double';
            
            output_data = Stateflow.Data(chart);
            output_data.Name = 'y';
            output_data.Scope = 'Output';
            output_data.DataType = 'double';
            
            fprintf('  ✅ Chart输入输出数据配置成功\n');
        end
        
        % 尝试手动连接
        try
            add_line(model_name, 'Input1/1', 'Test_Chart/1');
            fprintf('  ✅ 输入连接成功\n');
        catch
            fprintf('  ⚠️  输入连接失败，需要手动连接\n');
        end
        
        try
            add_line(model_name, 'Test_Chart/1', 'Output1/1');
            fprintf('  ✅ 输出连接成功\n');
        catch
            fprintf('  ⚠️  输出连接失败，需要手动连接\n');
        end
        
        % 保存
        save_system(model_name);
        open_system(model_name);
        
        fprintf('  ✅ 连接测试模型已打开，请检查连接状态\n');
        
        % 暂停以便查看
        pause(2);
        
        % 清理
        close_system(model_name, 0);
        delete([model_name '.slx']);
        
    catch ME
        fprintf('  ❌ 手动连接测试失败: %s\n', ME.message);
        if bdIsLoaded(model_name)
            close_system(model_name, 0);
        end
    end
end

% 执行测试
test_model_creation();