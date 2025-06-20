"""
简单的Simulink模型集成测试脚本
用于验证Python-Simulink接口是否正常工作
"""

import matlab.engine
import numpy as np


def test_simulink_integration():
    """测试Simulink模型的基本集成"""

    print("1. 启动MATLAB引擎...")
    eng = matlab.engine.start_matlab()

    try:
        # 检查模型文件
        model_name = 'sppvt_control_model'
        if eng.exist(f'{model_name}.slx', 'file') == 0:
            print(f"错误: 找不到模型文件 {model_name}.slx")
            return

        print("2. 加载Simulink模型...")
        eng.load_system(model_name, nargout=0)

        print("3. 准备输入数据...")
        # 创建简单的输入数据
        t = np.linspace(0, 0.05, 51)

        # 12个输入信号的常数值
        input_values = [
            1.0,  # error_value
            0.05,  # dt
            0.0,  # current_stage_offset
            1.5,  # sppvt_kp
            2.0,  # max_accel
            -3.0,  # max_decel
            0.8,  # prev_error
            4.0,  # prev_velocity
            -0.5,  # prev_accel
            0.05,  # sppvt_delta
            0.2,  # sppvt_eta
            1.0  # control_mode_flag
        ]

        # 创建输入矩阵
        input_data = []
        for time_val in t:
            row = [float(time_val)] + input_values
            input_data.append(row)

        # 转换为MATLAB数组
        eng.workspace['external_input_data'] = matlab.double(input_data)

        print("4. 配置模型参数...")
        eng.set_param(model_name, 'LoadExternalInput', 'on', nargout=0)
        eng.set_param(model_name, 'ExternalInput', '[external_input_data]', nargout=0)
        eng.set_param(model_name, 'SaveOutput', 'on', nargout=0)
        eng.set_param(model_name, 'OutputSaveName', 'yout', nargout=0)
        eng.set_param(model_name, 'SaveFormat', 'StructureWithTime', nargout=0)
        eng.set_param(model_name, 'StopTime', '0.05', nargout=0)

        print("5. 运行仿真...")
        # 方法1: 使用基本的sim命令
        eng.eval(f"sim('{model_name}');", nargout=0)

        print("6. 检查输出...")
        # 检查yout是否存在
        if eng.exist('yout', 'var'):
            print("✓ yout变量已创建")

            # 获取输出结构信息
            eng.eval("disp('yout结构:'); disp(yout);", nargout=0)

            # 尝试获取输出值
            try:
                # 获取信号数量
                num_signals = int(eng.eval("length(yout.signals)"))
                print(f"✓ 输出信号数量: {num_signals}")

                # 获取每个信号的最后值
                for i in range(1, min(6, num_signals + 1)):
                    last_value = float(eng.eval(f"yout.signals({i}).values(end)"))
                    print(f"  信号{i}最后值: {last_value:.6f}")

            except Exception as e:
                print(f"获取输出值时出错: {e}")

        else:
            print("✗ yout变量未创建")
            print("\n尝试方法2: 使用simOut返回值...")

            # 方法2: 捕获sim的返回值
            eng.eval(f"simOut = sim('{model_name}');", nargout=0)

            if eng.exist('simOut', 'var'):
                print("✓ simOut已创建")

                # 尝试不同的方式访问输出
                try:
                    # 方式1: 直接访问
                    eng.eval("yout = simOut.yout;", nargout=0)
                    print("✓ 通过simOut.yout获取输出")
                except:
                    try:
                        # 方式2: 使用get方法
                        eng.eval("yout = get(simOut, 'yout');", nargout=0)
                        print("✓ 通过get(simOut, 'yout')获取输出")
                    except:
                        # 方式3: 查看simOut的所有属性
                        print("simOut的属性:")
                        eng.eval("properties(simOut)", nargout=0)

        print("\n7. 清理...")
        eng.close_system(model_name, 0, nargout=0)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        eng.quit()
        print("\n测试完成")


def create_minimal_working_example():
    """创建一个最小的可工作示例"""

    print("创建最小可工作示例...\n")

    eng = matlab.engine.start_matlab()

    try:
        model_name = 'sppvt_control_model'

        # 加载模型
        eng.load_system(model_name, nargout=0)

        # 创建最简单的输入 - 单个时间点
        eng.eval("""
        % 清除工作空间
        clear all;

        % 创建输入数据 - 单个时间点
        external_input_data = [0, 1.0, 0.05, 0.0, 1.5, 2.0, -3.0, 0.8, 4.0, -0.5, 0.05, 0.2, 1.0];

        % 设置仿真参数
        set_param('sppvt_control_model', 'LoadExternalInput', 'on');
        set_param('sppvt_control_model', 'ExternalInput', '[external_input_data]');
        set_param('sppvt_control_model', 'StopTime', '0');
        set_param('sppvt_control_model', 'SaveOutput', 'on');
        set_param('sppvt_control_model', 'OutputSaveName', 'yout');
        set_param('sppvt_control_model', 'SaveFormat', 'StructureWithTime');

        % 运行仿真
        sim('sppvt_control_model');

        % 检查输出
        if exist('yout', 'var')
            disp('成功: yout已创建');
            disp(['输出信号数量: ' num2str(length(yout.signals))]);

            % 显示输出值
            for i = 1:length(yout.signals)
                disp(['信号' num2str(i) '值: ' num2str(yout.signals(i).values)]);
            end
        else
            disp('错误: yout未创建');
        end
        """, nargout=0)

        # 从Python访问结果
        if eng.exist('yout', 'var'):
            print("\nPython访问结果:")
            control_output = float(eng.eval("yout.signals(1).values"))
            print(f"控制输出: {control_output}")

        eng.close_system(model_name, 0, nargout=0)

    except Exception as e:
        print(f"错误: {e}")

    finally:
        eng.quit()


if __name__ == "__main__":
    print("=" * 50)
    print("Simulink集成测试")
    print("=" * 50)

    # 运行基本测试
    test_simulink_integration()

    print("\n" + "=" * 50)
    print("最小可工作示例")
    print("=" * 50)

    # 运行最小示例
    create_minimal_working_example()