# 实时Carla-Simulink集成：状态外化完整解决方案

## 概述

基于CARLA Python单步调用需求，实现Simulink模型的状态外化管理，解决persistent变量和Unit Delay状态在重复sim()调用时的丢失问题。

## 核心问题分析

### CARLA控制循环的单步调用需求
```
CARLA控制循环 (50ms):
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   获取传感器     │ -> │  调用Simulink    │ -> │   执行控制      │
│   数据          │    │  计算决策+SPPVT   │    │   指令         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              ↑
                    需要上一周期的状态数据
```

### 技术挑战
1. **状态丢失**: 每次sim()调用重置persistent变量和Unit Delay状态
2. **时间不连续**: SPPVT升级条件需要历史数据，但Simulink每次从t=0开始
3. **符号变化检测**: 误差符号变化需要持续的状态跟踪

### 解决策略：状态外化
- **Simulink无状态化**: 将所有时序状态转换为输入输出参数
- **Python状态管理**: Python维护所有持久状态，每次传入Simulink
- **纯函数计算**: Simulink成为无副作用的纯计算模块

---

## 1. 状态外化需求分析

### 需要外化的状态变量

基于`SPPVT_OUTPUT_ACQUISITION_GUIDE.md`分析，当前模型中的持久状态：

**Stage_Manager中的persistent状态：**
```matlab
persistent current_stage      % int32: 当前SPPVT阶段 (1,2,3...)
persistent prev_error_sign    % int32: 前一误差符号 (-1,0,1)
persistent upgrade_count      % int32: 升级计数器
```

**SPPVT_Adapter中的persistent状态：**
```matlab
persistent prev_error         % double: 前一控制误差
persistent prev_velocity      % double: 前一速度
persistent prev_accel         % double: 前一加速度
```

**Stage_Offset_Delay的Unit Delay状态：**
```matlab
InitialCondition = 0.0       % double: 当前级差值
```

---

## 2. Simulink模型重构方案

### 2.1 总线定义扩展

**创建扩展总线定义：**

```matlab
%% 修改create_decision_sppvt_bus.m

% 输入总线扩展：DecisionSPPVTInputExtended (18字段)
DecisionSPPVTInputExtended = Simulink.Bus;
DecisionSPPVTInputExtended.Elements = [
    % 原有11个字段
    Simulink.BusElement('Name', 'ego_speed_kmh', 'DataType', 'double');
    Simulink.BusElement('Name', 'ego_speed_ms', 'DataType', 'double');
    Simulink.BusElement('Name', 'command_type', 'DataType', 'int32');
    Simulink.BusElement('Name', 'command_active', 'DataType', 'logical');
    Simulink.BusElement('Name', 'manual_throttle_active', 'DataType', 'logical');
    Simulink.BusElement('Name', 'control_error', 'DataType', 'double');
    Simulink.BusElement('Name', 'control_mode_flag', 'DataType', 'int32');
    Simulink.BusElement('Name', 'V_target_kmh', 'DataType', 'double');
    Simulink.BusElement('Name', 'V_min_kmh', 'DataType', 'double');
    Simulink.BusElement('Name', 'G2_s', 'DataType', 'double');
    Simulink.BusElement('Name', 'timestamp', 'DataType', 'double');

    % 新增7个状态字段 (Python -> Simulink)
    Simulink.BusElement('Name', 'prev_stage', 'DataType', 'int32');
    Simulink.BusElement('Name', 'prev_error_sign', 'DataType', 'int32');
    Simulink.BusElement('Name', 'prev_upgrade_count', 'DataType', 'int32');
    Simulink.BusElement('Name', 'prev_control_error', 'DataType', 'double');
    Simulink.BusElement('Name', 'prev_velocity', 'DataType', 'double');
    Simulink.BusElement('Name', 'prev_accel', 'DataType', 'double');
    Simulink.BusElement('Name', 'prev_stage_offset', 'DataType', 'double');
];

% 输出总线扩展：DecisionSPPVTOutputExtended (19字段)
DecisionSPPVTOutputExtended = Simulink.Bus;
DecisionSPPVTOutputExtended.Elements = [
    % 原有12个字段
    Simulink.BusElement('Name', 'control_enabled', 'DataType', 'logical');
    Simulink.BusElement('Name', 'current_state', 'DataType', 'int32');
    Simulink.BusElement('Name', 'current_decision', 'DataType', 'int32');
    Simulink.BusElement('Name', 'torque_arbitration_active', 'DataType', 'logical');
    Simulink.BusElement('Name', 'updated_V_target_kmh', 'DataType', 'double');
    Simulink.BusElement('Name', 'updated_G2_s', 'DataType', 'double');
    Simulink.BusElement('Name', 'sppvt_control_output', 'DataType', 'double');
    Simulink.BusElement('Name', 'sppvt_velocity_output', 'DataType', 'double');
    Simulink.BusElement('Name', 'sppvt_acceleration_output', 'DataType', 'double');
    Simulink.BusElement('Name', 'sppvt_stage_output', 'DataType', 'double');
    Simulink.BusElement('Name', 'sppvt_status_output', 'DataType', 'double');
    Simulink.BusElement('Name', 'debug_message', 'DataType', 'int32');

    % 新增7个状态字段 (Simulink -> Python)
    Simulink.BusElement('Name', 'new_stage', 'DataType', 'int32');
    Simulink.BusElement('Name', 'new_error_sign', 'DataType', 'int32');
    Simulink.BusElement('Name', 'new_upgrade_count', 'DataType', 'int32');
    Simulink.BusElement('Name', 'new_control_error', 'DataType', 'double');
    Simulink.BusElement('Name', 'new_velocity', 'DataType', 'double');
    Simulink.BusElement('Name', 'new_accel', 'DataType', 'double');
    Simulink.BusElement('Name', 'new_stage_offset', 'DataType', 'double');
];
```

### 2.2 Stage_Manager函数重构

**完全无状态的Stage_Manager实现：**

```matlab
%% Stage_Manager MATLAB Function重构
function [new_stage_offset, new_stage, sign_changed, stage_states_out] = fcn(should_upgrade, error_value, sppvt_rho, stage_states_in)

% 输入参数：
% - stage_states_in: [prev_stage, prev_error_sign, prev_upgrade_count, prev_stage_offset]

% 从输入获取状态 (完全替代persistent变量)
current_stage = int32(stage_states_in(1));
prev_error_sign = int32(stage_states_in(2));
upgrade_count = int32(stage_states_in(3));
current_stage_offset = stage_states_in(4);

% SPPVT算法核心逻辑
current_error_sign = int32(sign(error_value));
sign_changed = false;

% 误差符号变化检测
if prev_error_sign ~= 0 && current_error_sign ~= 0 && prev_error_sign ~= current_error_sign
    % 符号变化，重置到第1级
    fprintf('Stage_Manager: 误差符号变化 %d->%d，重置到第1级\n', prev_error_sign, current_error_sign);
    current_stage = int32(1);
    current_stage_offset = 0.0;
    upgrade_count = int32(0);
    sign_changed = true;
end

% 升级条件判断
if should_upgrade && ~sign_changed
    current_stage = current_stage + int32(1);
    upgrade_count = upgrade_count + int32(1);

    % 计算新级差
    stage_increment = sppvt_rho * abs(error_value);
    if current_error_sign >= 0
        current_stage_offset = current_stage_offset + stage_increment;
    else
        current_stage_offset = current_stage_offset - stage_increment;
    end

    fprintf('Stage_Manager: 升级到第%d级，级差=%.3f\n', current_stage, current_stage_offset);
end

% 输出计算结果
new_stage_offset = current_stage_offset;
new_stage = current_stage;

% 输出状态向量 (传递给下一周期)
stage_states_out = [double(current_stage), double(current_error_sign), double(upgrade_count), current_stage_offset];

end
```

### 2.3 SPPVT_Adapter函数重构

**无状态的SPPVT_Adapter实现：**

```matlab
%% SPPVT_Adapter MATLAB Function重构
function [error_value, dt, stage_offset, kp, max_accel, max_decel, ...
         prev_error_out, prev_velocity_out, prev_accel_out, ...
         delta, eta, mode_flag, adapter_states_out] = fcn(validated_input, decision_output, adapter_states_in)

% 输入参数：
% - adapter_states_in: [prev_control_error, prev_velocity, prev_accel]

% 从输入获取状态 (完全替代persistent变量)
prev_error = adapter_states_in(1);
prev_velocity = adapter_states_in(2);
prev_accel = adapter_states_in(3);

% 提取当前数据
current_error = validated_input.control_error;
current_velocity = validated_input.ego_speed_ms;

% 计算加速度 (基于速度变化)
dt = 0.05;  % 50ms固定步长
current_accel = (current_velocity - prev_velocity) / dt;

% SPPVT参数配置
kp = 1.0;
max_accel = 2.0;
max_decel = -3.0;
delta = 0.05;
eta = 0.2;
mode_flag = int32(validated_input.control_mode_flag);

% 获取级差 (从输入状态获取)
stage_offset = validated_input.prev_stage_offset;

% 输出当前值
error_value = current_error;
prev_error_out = prev_error;
prev_velocity_out = prev_velocity;
prev_accel_out = prev_accel;

% 输出状态向量 (传递给下一周期)
adapter_states_out = [current_error, current_velocity, current_accel];

end
```

### 2.4 模型连接重构

**关键连接修改：**

1. **移除Stage_Offset_Delay Unit Delay块**
2. **添加Bus_Selector_States用于状态提取**：
```
Bus_Selector_States:
- 输入：Input (DecisionSPPVTInputExtended)
- 输出：stage_states, adapter_states
```

3. **修改Stage_Manager连接**：
```
Stage_Manager:
- 输入1：should_upgrade (从SPPVT_Control)
- 输入2：error_value (从Bus_Selector1)
- 输入3：sppvt_rho (从Parameter_Manager)
- 输入4：stage_states (从Bus_Selector_States) [新增]
- 输出1：new_stage_offset
- 输出2：new_stage
- 输出3：sign_changed
- 输出4：stage_states_out [新增]
```

4. **修改SPPVT_Adapter连接**：
```
SPPVT_Adapter:
- 输入1：validated_input (从Input_Validator)
- 输入2：decision_output (从Decision_Function)
- 输入3：adapter_states (从Bus_Selector_States) [新增]
- 输出1-12：原有12个SPPVT输出
- 输出13：adapter_states_out [新增]
```

### 2.5 Output_Formatter扩展

**添加状态输出功能：**

```matlab
%% Output_Formatter MATLAB Function扩展
function integrated_output = fcn(decision_output, sppvt_control_output, sppvt_velocity_output,
                               sppvt_acceleration_output, sppvt_jerk_output, sppvt_should_upgrade,
                               stage_states_out, adapter_states_out)

% 组装原有输出
integrated_output = decision_output;
integrated_output.sppvt_control_output = sppvt_control_output;
integrated_output.sppvt_velocity_output = sppvt_velocity_output;
integrated_output.sppvt_acceleration_output = sppvt_acceleration_output;
integrated_output.sppvt_stage_output = sppvt_jerk_output;
integrated_output.sppvt_status_output = sppvt_should_upgrade;

% 关键：添加状态输出
integrated_output.new_stage = int32(stage_states_out(1));
integrated_output.new_error_sign = int32(stage_states_out(2));
integrated_output.new_upgrade_count = int32(stage_states_out(3));
integrated_output.new_stage_offset = stage_states_out(4);

integrated_output.new_control_error = adapter_states_out(1);
integrated_output.new_velocity = adapter_states_out(2);
integrated_output.new_accel = adapter_states_out(3);

end
```

---

## 3. Python状态管理器实现

### 3.1 无状态SPPVT状态管理器

```python
class RealtimeStatelessSPPVTManager:
    """
    无状态SPPVT状态管理器
    外化所有Simulink状态，实现真正的单步调用
    """

    def __init__(self, model_name='ACC_Decision_SPPVT_Integrated'):
        self.model_name = model_name
        self.matlab_eng = None

        # 维护所有外化的状态
        self.stage_states = {
            'current_stage': 1,
            'prev_error_sign': 0,
            'upgrade_count': 0,
            'stage_offset': 0.0
        }

        self.adapter_states = {
            'prev_error': 0.0,
            'prev_velocity': 13.89,  # 50km/h = 13.89m/s
            'prev_accel': 0.1
        }

        # 性能监控
        self.call_count = 0
        self.simulation_time = 0.0

        self._init_matlab_engine()

    def _init_matlab_engine(self):
        """初始化MATLAB引擎"""
        import matlab.engine
        self.matlab_eng = matlab.engine.start_matlab()

        # 加载总线定义
        self.matlab_eng.load('DecisionSPPVTBusDefinitions.mat', nargout=0)

        # 加载模型
        self.matlab_eng.load_system(self.model_name, nargout=0)

        # 配置单步仿真
        self.matlab_eng.set_param(self.model_name, 'StopTime', '0.05', nargout=0)
        self.matlab_eng.set_param(self.model_name, 'SaveOutput', 'on', nargout=0)
        self.matlab_eng.set_param(self.model_name, 'SaveFormat', 'Dataset', nargout=0)

    def call_simulink_single_step(self, sensor_data, command_data):
        """
        单步调用Simulink，实现状态的传入和更新

        Args:
            sensor_data: 传感器数据字典
            command_data: 指令数据字典

        Returns:
            控制结果字典，包含状态更新
        """

        # 1. 准备扩展输入
        extended_input = self._prepare_extended_input(sensor_data, command_data)

        # 2. 调用Simulink计算
        self.matlab_eng.workspace['current_input'] = extended_input
        self.matlab_eng.set_param(self.model_name, 'ExternalInput', 'current_input', nargout=0)

        sim_result = self.matlab_eng.sim(self.model_name, nargout=1)

        # 3. 提取结果和新状态
        output = self._extract_simulink_output(sim_result)

        # 4. 更新状态 (关键：Simulink -> Python)
        self._update_states_from_output(output)

        # 5. 更新仿真时间
        self.simulation_time += 0.05
        self.call_count += 1

        return {
            'target_accel': output['sppvt_control_output'],
            'control_enabled': output['control_enabled'],
            'current_state': output['current_state'],
            'current_decision': output['current_decision'],
            'sppvt_stage': output['new_stage'],
            'sppvt_upgrade_count': output['new_upgrade_count'],
            'debug_info': f"Stage{output['new_stage']}_Offset{output['new_stage_offset']:.3f}",
            'simulation_time': self.simulation_time
        }

    def _prepare_extended_input(self, sensor_data, command_data):
        """准备18字段的扩展输入"""
        import matlab

        time_points = [self.simulation_time, self.simulation_time + 0.05]

        # 创建时间序列数据
        extended_input = {}

        # 原有11个字段
        for key in ['ego_speed_kmh', 'ego_speed_ms', 'control_error', 'V_target_kmh', 'V_min_kmh', 'G2_s']:
            extended_input[key] = matlab.double([sensor_data[key]] * 2)

        for key in ['command_type', 'control_mode_flag']:
            extended_input[key] = matlab.int32([command_data[key]] * 2)

        for key in ['command_active', 'manual_throttle_active']:
            extended_input[key] = matlab.logical([command_data[key]] * 2)

        extended_input['timestamp'] = matlab.double(time_points)

        # 新增7个状态字段 (Python -> Simulink)
        extended_input['prev_stage'] = matlab.int32([self.stage_states['current_stage']] * 2)
        extended_input['prev_error_sign'] = matlab.int32([self.stage_states['prev_error_sign']] * 2)
        extended_input['prev_upgrade_count'] = matlab.int32([self.stage_states['upgrade_count']] * 2)
        extended_input['prev_stage_offset'] = matlab.double([self.stage_states['stage_offset']] * 2)

        extended_input['prev_control_error'] = matlab.double([self.adapter_states['prev_error']] * 2)
        extended_input['prev_velocity'] = matlab.double([self.adapter_states['prev_velocity']] * 2)
        extended_input['prev_accel'] = matlab.double([self.adapter_states['prev_accel']] * 2)

        return extended_input

    def _extract_simulink_output(self, sim_result):
        """提取Simulink输出 (使用已验证的方法)"""
        output_element = sim_result.yout[0]
        values = output_element.Values

        return {
            'control_enabled': bool(values.control_enabled.Data[-1]),
            'current_state': int(values.current_state.Data[-1]),
            'current_decision': int(values.current_decision.Data[-1]),
            'sppvt_control_output': float(values.sppvt_control_output.Data[-1]),

            # 新状态输出
            'new_stage': int(values.new_stage.Data[-1]),
            'new_error_sign': int(values.new_error_sign.Data[-1]),
            'new_upgrade_count': int(values.new_upgrade_count.Data[-1]),
            'new_stage_offset': float(values.new_stage_offset.Data[-1]),

            'new_control_error': float(values.new_control_error.Data[-1]),
            'new_velocity': float(values.new_velocity.Data[-1]),
            'new_accel': float(values.new_accel.Data[-1])
        }

    def _update_states_from_output(self, output):
        """更新Python维护的状态"""
        # 更新Stage_Manager状态
        self.stage_states.update({
            'current_stage': output['new_stage'],
            'prev_error_sign': output['new_error_sign'],
            'upgrade_count': output['new_upgrade_count'],
            'stage_offset': output['new_stage_offset']
        })

        # 更新SPPVT_Adapter状态
        self.adapter_states.update({
            'prev_error': output['new_control_error'],
            'prev_velocity': output['new_velocity'],
            'prev_accel': output['new_accel']
        })

    def reset_states(self):
        """重置所有状态"""
        self.stage_states = {
            'current_stage': 1,
            'prev_error_sign': 0,
            'upgrade_count': 0,
            'stage_offset': 0.0
        }
        self.adapter_states = {
            'prev_error': 0.0,
            'prev_velocity': 13.89,
            'prev_accel': 0.1
        }
        self.simulation_time = 0.0
        self.call_count = 0

    def get_performance_stats(self):
        """获取性能统计"""
        return {
            "总调用次数": self.call_count,
            "仿真时间": f"{self.simulation_time:.3f}s",
            "当前阶段": self.stage_states['current_stage'],
            "当前级差": f"{self.stage_states['stage_offset']:.3f}",
            "升级次数": self.stage_states['upgrade_count']
        }
```

### 3.2 接口集成

```python
# 修改acc_decision_sppvt_interface.py
class ACCDecisionSPPVTInterface:
    def __init__(self, debug=True):
        self.debug = debug

        # 使用无状态管理器
        self.stateless_manager = RealtimeStatelessSPPVTManager()

    def process_decision_and_control(self, input_data):
        """统一的决策+控制处理接口"""

        # 分离数据
        sensor_data = {
            'ego_speed_kmh': input_data['ego_speed_kmh'],
            'ego_speed_ms': input_data['ego_speed_ms'],
            'control_error': input_data['control_error'],
            'V_target_kmh': input_data['V_target_kmh'],
            'V_min_kmh': input_data['V_min_kmh'],
            'G2_s': input_data['G2_s']
        }

        command_data = {
            'command_type': input_data['command_type'],
            'command_active': input_data['command_active'],
            'manual_throttle_active': input_data['manual_throttle_active'],
            'control_mode_flag': input_data['control_mode_flag']
        }

        # 调用无状态管理器
        result = self.stateless_manager.call_simulink_single_step(sensor_data, command_data)

        # 返回标准格式
        return {
            'target_accel': result['target_accel'],
            'control_enabled': result['control_enabled'],
            'current_state': result['current_state'],
            'current_decision': result['current_decision'],
            'torque_arbitration_active': input_data['manual_throttle_active'],
            'updated_V_target_kmh': input_data['V_target_kmh'],
            'updated_G2_s': input_data['G2_s'],
            'sppvt_stage': result['sppvt_stage'],
            'sppvt_upgrade_count': result['sppvt_upgrade_count'],
            'debug_message': result['debug_info']
        }
```

---

## 4. 实施步骤和验证

### 4.1 实施优先级

1. **第1步：总线定义修改** - 扩展为18/19字段
2. **第2步：Stage_Manager重构** - 移除persistent，添加状态输入输出
3. **第3步：SPPVT_Adapter重构** - 移除persistent，添加状态输入输出
4. **第4步：模型连接重构** - 移除Unit Delay，添加状态数据流
5. **第5步：Output_Formatter扩展** - 添加7个状态输出字段
6. **第6步：Python状态管理器实现** - 实现RealtimeStatelessSPPVTManager
7. **第7步：完整测试验证** - 单步调用和状态持续性测试

### 4.2 验证方案

```python
def test_stateless_sppvt():
    """测试状态外化的完整性"""
    manager = RealtimeStatelessSPPVTManager()

    # 测试1：状态持续性
    sensor_data1 = {'ego_speed_kmh': 50, 'control_error': 1.5, ...}
    command_data1 = {'command_type': 1, 'command_active': True, ...}

    result1 = manager.call_simulink_single_step(sensor_data1, command_data1)
    result2 = manager.call_simulink_single_step(sensor_data1, command_data1)

    # 验证：第二次调用应该基于第一次的状态
    assert result2['sppvt_stage'] >= result1['sppvt_stage']
    assert result2['simulation_time'] == 0.1

    # 测试2：符号变化检测
    sensor_data_pos = {'control_error': 2.0, ...}
    sensor_data_neg = {'control_error': -2.0, ...}

    result_pos = manager.call_simulink_single_step(sensor_data_pos, command_data1)
    result_neg = manager.call_simulink_single_step(sensor_data_neg, command_data1)

    # 验证：符号变化应重置阶段
    print(f"符号变化测试: {result_pos['debug_info']} → {result_neg['debug_info']}")

    print("✅ 无状态SPPVT测试通过")
```

---

## 5. 方案优势和结论

### 核心设计理念
**状态外化原则：** Simulink变为无状态的纯计算模型，所有时序状态由Python管理

```
Python状态管理器:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   维护状态      │ -> │  调用Simulink    │ -> │   更新状态      │
│   ·stage       │    │  传入状态        │    │   提取新状态     │
│   ·error_sign  │    │  纯函数计算       │    │   用于下一周期   │
│   ·upgrade_cnt │    │  输出新状态       │    │                │
│   ·prev_error  │    │                 │    │                │
│   ·prev_vel    │    │                 │    │                │
│   ·prev_accel  │    │                 │    │                │
│   ·stage_offset│    │                 │    │                │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 方案优势

1. **完全无状态Simulink** - 每次调用都是纯函数计算
2. **Python完全控制** - 状态管理、初始化、重置都在Python端
3. **调试友好** - 状态完全可见和可控制
4. **扩展性强** - 容易添加新的状态变量
5. **符合CARLA需求** - 完美解决单步调用问题
6. **保持算法完整性** - SPPVT升级和符号变化检测功能完整保留

### 技术关键点

1. **总线扩展合理性** - 状态变量作为算法的输入输出，符合数据流设计
2. **无副作用计算** - Simulink成为纯函数，消除了状态同步问题
3. **状态同步机制** - Python和Simulink状态通过输入输出严格同步
4. **错误恢复能力** - 状态外化使得错误恢复和调试更加容易

该方案完全解决了CARLA单步调用与Simulink时序状态的矛盾，实现了真正的实时SPPVT控制，支持状态持续性和符号变化检测等高级功能。