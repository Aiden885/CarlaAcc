"""
实时SPPVT状态管理器
解决Simulink重复调用sim()时的状态持久化问题
高内聚低耦合设计，专门管理SPPVT相关的所有状态
"""

import numpy as np
import matlab.engine
import matlab
from typing import Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class SPPVTState:
    """SPPVT状态数据结构"""
    current_stage_offset: float = 0.0
    prev_error: float = 0.0
    prev_velocity: float = 13.89  # 50 km/h = 13.89 m/s
    prev_accel: float = 0.1
    last_update_time: float = 0.0

    # 历史记录用于分析
    error_history: deque = None
    velocity_history: deque = None
    accel_history: deque = None

    def __post_init__(self):
        if self.error_history is None:
            self.error_history = deque(maxlen=100)
        if self.velocity_history is None:
            self.velocity_history = deque(maxlen=100)
        if self.accel_history is None:
            self.accel_history = deque(maxlen=100)

@dataclass
class SimulinkInputs:
    """Simulink输入数据结构"""
    control_error: float
    ego_speed_ms: float
    control_mode_flag: bool
    control_enabled: bool

    # 外部状态输入
    external_stage_offset: float = 0.0
    external_prev_error: float = 0.0
    external_prev_velocity: float = 13.89
    external_prev_accel: float = 0.1

class RealtimeSPPVTStateManager:
    """
    实时SPPVT状态管理器

    核心功能:
    1. 管理SPPVT算法的所有持久化状态
    2. 提供单步仿真接口，避免状态重置
    3. 实现外部状态注入，确保状态连续性
    4. 性能监控和异常处理
    """

    def __init__(self, matlab_engine=None, model_name='ACC_Decision_SPPVT_Integrated'):
        self.logger = logging.getLogger(__name__)

        # MATLAB引擎管理
        self.matlab_eng = matlab_engine
        self.model_name = model_name
        self.model_loaded = False

        # 状态管理
        self.sppvt_state = SPPVTState()
        self.simulation_time = 0.0
        self.dt = 0.05  # 50ms时间步长

        # 性能监控
        self.execution_times = deque(maxlen=1000)
        self.error_count = 0
        self.last_successful_output = None

        # 配置参数
        self.config = {
            'sppvt_kp': 1.0,
            'max_accel': 2.0,
            'max_decel': -3.0,
            'sppvt_delta': 0.05,
            'sppvt_eta': 0.2,
            'timeout_seconds': 1.0,
            'max_consecutive_errors': 5
        }

        self._initialize_matlab_engine()

    def _initialize_matlab_engine(self):
        """初始化MATLAB引擎"""
        try:
            if self.matlab_eng is None:
                self.logger.info("启动MATLAB引擎...")
                self.matlab_eng = matlab.engine.start_matlab()
                self.logger.info("MATLAB引擎启动成功")

            # 加载模型
            self._load_simulink_model()

        except Exception as e:
            self.logger.error(f"MATLAB引擎初始化失败: {e}")
            raise

    def _load_simulink_model(self):
        """加载Simulink模型"""
        try:
            self.logger.info(f"Loading Simulink model: {self.model_name}")

            # 首先创建必需的总线定义
            self.logger.info("Creating 14/15-field bus definitions...")
            self.matlab_eng.eval("create_decision_sppvt_bus()", nargout=0)

            # 检查总线是否成功创建
            bus_check = self.matlab_eng.eval("exist('DecisionSPPVTInputExtended', 'var')")
            if bus_check == 0:
                self.logger.error("DecisionSPPVTInputExtended bus creation failed")
                raise RuntimeError("Bus definition creation failed")

            self.logger.info("Success: 14/15-field bus definitions created")

            # 加载模型
            self.matlab_eng.eval(f"load_system('{self.model_name}')", nargout=0)

            # 配置仿真参数 - 参考两种成功的实现方式
            self.matlab_eng.eval(f"set_param('{self.model_name}', 'SimulationMode', 'normal')", nargout=0)
            self.matlab_eng.eval(f"set_param('{self.model_name}', 'StopTime', '{self.dt}')", nargout=0)
            self.matlab_eng.eval(f"set_param('{self.model_name}', 'SaveOutput', 'on')", nargout=0)
            self.matlab_eng.eval(f"set_param('{self.model_name}', 'OutputSaveName', 'yout')", nargout=0)

            # 强制使用Dataset格式 - StructureWithTime不支持总线数据记录
            # 根据Simulink文档：总线数据输出必须使用Dataset格式
            self.matlab_eng.eval(f"set_param('{self.model_name}', 'SaveFormat', 'Dataset')", nargout=0)
            self.output_format = 'Dataset'
            self.logger.info("Configured Simulink output format: Dataset (required for bus data)")

            # 清除任何旧的外部输入配置
            try:
                self.matlab_eng.eval(f"set_param('{self.model_name}', 'ExternalInput', '')", nargout=0)
                self.matlab_eng.eval(f"set_param('{self.model_name}', 'LoadExternalInput', 'off')", nargout=0)
                self.logger.info("Cleared external input configuration")
            except:
                pass  # 忽略配置错误

            self.model_loaded = True
            self.logger.info("Simulink model loaded successfully")

        except Exception as e:
            self.logger.error(f"Simulink model loading failed: {e}")
            # 如果是总线相关错误，提供详细信息
            if "Bus" in str(e) or "DecisionSPPVT" in str(e):
                self.logger.error("This might be a bus definition issue. Please check:")
                self.logger.error("1. create_decision_sppvt_bus.m file exists")
                self.logger.error("2. Simulink model uses correct bus names")
                self.logger.error("3. Inport/Outport configured for new 14/15-field buses")
            raise

    def update_state_from_simulink_output(self, sim_out):
        """从Simulink输出更新状态 - 使用正确的Dataset格式访问"""
        try:
            # 使用与test_integrated_sppvt_model.m一致的解析逻辑
            # 检查Simulink.SimulationOutput格式
            if hasattr(sim_out, 'yout') and sim_out.yout is not None:
                output_data = sim_out.yout

                # 检查Dataset格式
                if hasattr(output_data, 'numElements') and output_data.numElements >= 1:
                    element = self.matlab_eng.eval("sim_out.yout{1}")

                    # 检查是否为Signal格式
                    if hasattr(element, 'Values') and element.Values is not None:
                        values_struct = element.Values

                        # 提取SPPVT控制输出
                        sppvt_output = self.matlab_eng.eval(
                            "sim_out.yout{1}.Values.sppvt_control_output.Data(end)"
                        )

                        # 提取新的状态外化字段 (15-field版本的新字段)
                        try:
                            # new_stage_offset (第13个字段)
                            new_stage_offset = self.matlab_eng.eval(
                                "sim_out.yout{1}.Values.new_stage_offset.Data(end)"
                            )
                            self.sppvt_state.current_stage_offset = float(new_stage_offset)
                            self.logger.debug(f"Updated stage_offset: {new_stage_offset}")
                        except Exception as e:
                            self.logger.debug(f"Could not extract new_stage_offset: {e}")

                        try:
                            # new_adapter_states (第15个字段) - [control_error, velocity, accel]
                            new_adapter_states = self.matlab_eng.eval(
                                "sim_out.yout{1}.Values.new_adapter_states.Data(end,:)"
                            )
                            if hasattr(new_adapter_states, '__len__') and len(new_adapter_states) >= 3:
                                self.sppvt_state.prev_error = float(new_adapter_states[0])
                                self.sppvt_state.prev_velocity = float(new_adapter_states[1])
                                self.sppvt_state.prev_accel = float(new_adapter_states[2])
                                self.logger.debug(f"Updated adapter states: {new_adapter_states}")
                        except Exception as e:
                            self.logger.debug(f"Could not extract new_adapter_states: {e}")

                        # 更新时间戳
                        self.sppvt_state.last_update_time = time.time()

                        # 记录历史数据
                        self.sppvt_state.error_history.append(self.sppvt_state.prev_error)
                        self.sppvt_state.velocity_history.append(self.sppvt_state.prev_velocity)
                        self.sppvt_state.accel_history.append(self.sppvt_state.prev_accel)

                        return float(sppvt_output)
                    else:
                        raise ValueError("Dataset element does not have Values struct")
                else:
                    raise ValueError("Dataset format invalid or empty")
            else:
                raise ValueError("sim_out does not have yout attribute")

        except Exception as e:
            self.logger.error(f"Failed to update state from Simulink output: {e}")
            # 添加详细的调试信息
            try:
                sim_out_type = str(type(sim_out))
                self.logger.debug(f"sim_out type: {sim_out_type}")
                if hasattr(sim_out, 'yout'):
                    yout_type = str(type(sim_out.yout))
                    self.logger.debug(f"yout type: {yout_type}")
                    if hasattr(sim_out.yout, 'numElements'):
                        self.logger.debug(f"Dataset numElements: {sim_out.yout.numElements}")
            except:
                pass
            return None

    def update_state_from_simulink_output_eval(self, current_control_error: float, current_ego_speed_ms: float):
        """使用eval方式从Simulink输出更新状态 - 参考sppvt_longitudinal_control.py"""
        try:
            # 从simOut获取yout - 使用eval方式访问
            self.matlab_eng.eval("yout = simOut.yout;", nargout=0)

            # 确认yout存在
            if not self.matlab_eng.eval("exist('yout', 'var')"):
                # 尝试其他方式
                self.matlab_eng.eval("yout = get(simOut, 'yout');", nargout=0)

            # 调试Dataset结构 - 先检查yout的实际结构
            try:
                # 检查yout类型和结构
                yout_class = self.matlab_eng.eval("class(yout)")
                self.logger.debug(f"yout class: {yout_class}")

                is_dataset = self.matlab_eng.eval("isa(yout, 'Simulink.SimulationData.Dataset')")
                self.logger.debug(f"Is Dataset: {is_dataset}")

                if is_dataset:
                    # 检查Dataset元素数量
                    num_elements = int(self.matlab_eng.eval("yout.numElements"))
                    self.logger.debug(f"Dataset has {num_elements} elements")

                    if num_elements > 0:
                        # 检查第一个元素的结构
                        element_class = self.matlab_eng.eval("class(yout{1})")
                        element_props = self.matlab_eng.eval("properties(yout{1})")
                        self.logger.debug(f"Element 1 class: {element_class}")
                        self.logger.debug(f"Element 1 properties: {element_props}")

                        # 尝试不同的Dataset访问方式
                        if self.matlab_eng.eval("isprop(yout{1}, 'Values')"):
                            # 有Values属性 - 这是总线信号
                            values_class = self.matlab_eng.eval("class(yout{1}.Values)")
                            self.logger.debug(f"Values class: {values_class}")

                            # 对于总线信号，Values是一个结构体，包含各个字段
                            if self.matlab_eng.eval("isstruct(yout{1}.Values)"):
                                # Values是结构体，包含各个总线字段
                                # 使用MATLAB eval直接获取字段名称
                                self.logger.debug("Values is a struct, getting field names")

                                # 处理timeseries对象 - Dataset格式中每个字段都是timeseries
                                try:
                                    # 获取SPPVT控制输出 - 从timeseries对象获取数据
                                    sppvt_output_raw = self.matlab_eng.eval("yout{1}.Values.sppvt_control_output.Data(end)")
                                    sppvt_output = float(sppvt_output_raw)
                                    self.logger.debug(f"Successfully extracted sppvt_control_output from timeseries: {sppvt_output}")

                                    # 提取状态外化字段
                                    try:
                                        new_stage_offset_raw = self.matlab_eng.eval("yout{1}.Values.new_stage_offset.Data(end)")
                                        new_stage_offset = float(new_stage_offset_raw)
                                        self.sppvt_state.current_stage_offset = new_stage_offset
                                        self.logger.debug(f"Updated stage_offset from timeseries: {new_stage_offset}")

                                        # 提取adapter状态 - 处理可能的不同数据结构
                                        try:
                                            # 首先检查数据维度
                                            data_size = self.matlab_eng.eval("size(yout{1}.Values.new_adapter_states.Data)")
                                            self.logger.debug(f"new_adapter_states Data size: {data_size}")

                                            # 检查数据内容和格式
                                            raw_data = self.matlab_eng.eval("yout{1}.Values.new_adapter_states.Data")
                                            self.logger.debug(f"Raw new_adapter_states data: {raw_data}")

                                            # 正确处理嵌套的MATLAB数组结构
                                            # raw_data 格式: [[[val1]], [[val2]], [[val3]]] - 3个状态，每个都是时间序列数组
                                            if hasattr(raw_data, '__len__') and len(raw_data) >= 3:
                                                try:
                                                    # 直接从raw_data中提取数据，避免索引错误
                                                    # raw_data结构: [[[val1]], [[val2]], [[val3]]]
                                                    # 我们需要提取每个嵌套数组中的第一个值
                                                    if len(raw_data) >= 3:
                                                        # 尝试提取嵌套数组中的值
                                                        state1_raw = raw_data[0]  # 第一个状态 [[control_error_values]]
                                                        state2_raw = raw_data[1]  # 第二个状态 [[velocity_values]]
                                                        state3_raw = raw_data[2]  # 第三个状态 [[acceleration_values]]

                                                        # 从嵌套结构中提取实际值
                                                        if (hasattr(state1_raw, '__len__') and len(state1_raw) > 0 and
                                                            hasattr(state1_raw[0], '__len__') and len(state1_raw[0]) > 0):
                                                            # 获取最后一个时间点的值
                                                            state1 = float(state1_raw[0][-1])  # control_error
                                                            state2 = float(state2_raw[0][-1])  # velocity
                                                            state3 = float(state3_raw[0][-1])  # acceleration

                                                            # 检查是否是固定的维度信息 [3.0, 1.0, 2.0]
                                                            if (abs(state1 - 3.0) < 0.001 and abs(state2 - 1.0) < 0.001 and abs(state3 - 2.0) < 0.001):
                                                                self.logger.debug("Detected dimension info [3,1,2], using current input values")
                                                                # 维度信息，不是真实状态，但我们仍然需要状态连续性
                                                                # 先保持当前状态不变，使用函数参数更新
                                                                adapter_values = [current_control_error, current_ego_speed_ms, 0.0]
                                                            else:
                                                                # 这是真实的状态数据
                                                                adapter_values = [state1, state2, state3]
                                                                self.logger.debug(f"Successfully extracted real state data: {adapter_values}")
                                                        else:
                                                            raise ValueError("Cannot access nested state data")
                                                    else:
                                                        raise ValueError("Insufficient state data elements")

                                                except Exception as extract_error:
                                                    self.logger.debug(f"State extraction failed: {extract_error}, using fallback")
                                                    # 使用函数参数作为fallback，这是最可靠的方式
                                                    adapter_values = [current_control_error, current_ego_speed_ms, 0.0]
                                            else:
                                                # 数组长度不足，使用函数参数
                                                self.logger.debug("Insufficient data length, using input values")
                                                adapter_values = [current_control_error, current_ego_speed_ms, 0.0]

                                            self.sppvt_state.prev_error = adapter_values[0]
                                            self.sppvt_state.prev_velocity = adapter_values[1]
                                            self.sppvt_state.prev_accel = adapter_values[2]
                                            self.logger.debug(f"Updated adapter states from timeseries: {adapter_values}")

                                        except Exception as adapter_error:
                                            self.logger.debug(f"Adapter state extraction failed: {adapter_error}, using current input values")
                                            # 作为备选方案，使用当前输入值来维护基本状态
                                            # 注意：这里需要使用函数参数值而不是局部变量
                                            pass  # 状态维护将在外层处理
                                    except Exception as e:
                                        self.logger.debug(f"Could not extract state fields from timeseries: {e}")

                                except Exception as field_error:
                                    self.logger.debug(f"Timeseries access failed: {field_error}")
                                    # 备用方案：尝试其他字段的timeseries数据
                                    try:
                                        # 尝试速度输出的timeseries数据
                                        sppvt_output = float(self.matlab_eng.eval("yout{1}.Values.sppvt_velocity_output.Data(end)"))
                                        self.logger.debug(f"Using sppvt_velocity_output timeseries as fallback: {sppvt_output}")
                                    except:
                                        try:
                                            # 尝试加速度输出的timeseries数据
                                            sppvt_output = float(self.matlab_eng.eval("yout{1}.Values.sppvt_acceleration_output.Data(end)"))
                                            self.logger.debug(f"Using sppvt_acceleration_output timeseries as fallback: {sppvt_output}")
                                        except:
                                            sppvt_output = 0.0
                                            self.logger.warning("Could not extract any timeseries data, using default 0.0")

                                except Exception as e:
                                    self.logger.debug(f"Struct field access failed: {e}, trying generic approach")
                                    # 备用方案：获取所有字段值并选择合适的输出
                                    try:
                                        # 方法1：尝试直接访问结构体数据
                                        field_names_str = self.matlab_eng.eval("strjoin(fieldnames(yout{1}.Values), ',')")
                                        self.logger.debug(f"Struct fields: {field_names_str}")

                                        # 方法2：尝试cell2mat转换
                                        values_array = self.matlab_eng.eval("cell2mat(struct2cell(yout{1}.Values))")
                                        self.logger.debug(f"Values array shape: {self.matlab_eng.eval('size(cell2mat(struct2cell(yout{{1}}.Values)))')}")

                                        # 假设第7个字段是sppvt_control_output (索引从1开始)
                                        if self.matlab_eng.eval("numel(cell2mat(struct2cell(yout{1}.Values))) >= 7"):
                                            sppvt_output = float(self.matlab_eng.eval("cell2mat(struct2cell(yout{1}.Values))(7)"))
                                            self.logger.debug(f"Using cell2mat approach: {sppvt_output}")
                                        else:
                                            # 使用第一个字段
                                            sppvt_output = float(self.matlab_eng.eval("cell2mat(struct2cell(yout{1}.Values))(1)"))
                                            self.logger.debug(f"Using first field: {sppvt_output}")
                                    except Exception as e2:
                                        self.logger.debug(f"Generic approach also failed: {e2}")
                                        sppvt_output = 0.0

                            else:
                                # Values不是结构体，尝试数组访问
                                try:
                                    sppvt_output = float(self.matlab_eng.eval("yout{1}.Values(end, 1)"))
                                except:
                                    sppvt_output = float(self.matlab_eng.eval("yout{1}.Values(end)"))

                        else:
                            # 没有Values属性，直接访问Dataset元素
                            if self.matlab_eng.eval("isnumeric(yout{1})"):
                                sppvt_output = float(self.matlab_eng.eval("yout{1}(end)"))
                            else:
                                sppvt_output = float(self.matlab_eng.eval("double(yout{1})"))

                        self.logger.debug(f"Extracted SPPVT output: {sppvt_output}")

                else:
                    # 非Dataset格式 - StructureWithTime
                    if self.matlab_eng.eval("isfield(yout, 'signals')"):
                        sppvt_output = float(self.matlab_eng.eval("yout.signals(1).values(end)"))
                    else:
                        sppvt_output = float(self.matlab_eng.eval("yout(end, 1)"))

            except Exception as e:
                self.logger.error(f"Dataset access failed: {e}")
                # 获取更详细的错误信息
                try:
                    self.matlab_eng.eval("disp('=== DEBUGGING YOUT STRUCTURE ===');", nargout=0)
                    self.matlab_eng.eval("disp(['yout class: ', class(yout)]);", nargout=0)
                    if self.matlab_eng.eval("isa(yout, 'Simulink.SimulationData.Dataset')"):
                        self.matlab_eng.eval("disp(['Dataset elements: ', num2str(yout.numElements)]);", nargout=0)
                        if int(self.matlab_eng.eval("yout.numElements")) > 0:
                            self.matlab_eng.eval("disp(['Element 1 class: ', class(yout{1})]);", nargout=0)
                            self.matlab_eng.eval("disp('Element 1 properties:'); disp(properties(yout{1}));", nargout=0)
                except:
                    pass

                sppvt_output = 0.0  # 默认值
                self.logger.warning("Could not extract SPPVT output, using default value")

            # 更新时间戳和历史记录
            self.sppvt_state.last_update_time = time.time()
            self.sppvt_state.error_history.append(self.sppvt_state.prev_error)
            self.sppvt_state.velocity_history.append(self.sppvt_state.prev_velocity)
            self.sppvt_state.accel_history.append(self.sppvt_state.prev_accel)

            return sppvt_output

        except Exception as e:
            self.logger.error(f"Failed to update state from Simulink output (eval): {e}")
            return None

    def prepare_simulink_inputs(self, control_error: float, ego_speed_ms: float,
                              control_mode_flag: bool, control_enabled: bool) -> Dict[str, Any]:
        """准备Simulink输入，注入外部状态到总线结构 - 14-field版本"""

        # 更新状态历史
        self.sppvt_state.prev_error = control_error if control_enabled else 0.0
        self.sppvt_state.prev_velocity = ego_speed_ms

        ego_speed_kmh = ego_speed_ms * 3.6

        # 准备总线输入结构 - DecisionSPPVTInputExtended (14字段)
        simulink_inputs = {
            # 原有的11个字段
            'ego_speed_kmh': matlab.double([ego_speed_kmh]),
            'ego_speed_ms': matlab.double([ego_speed_ms]),
            'command_type': matlab.int32([1]),  # 默认I0指令
            'command_active': matlab.logical([True]),
            'manual_throttle_active': matlab.logical([False]),
            'control_error': matlab.double([control_error]),
            'control_mode_flag': matlab.int32([1 if control_mode_flag else 2]),
            'V_target_kmh': matlab.double([50.0]),  # 默认目标速度
            'V_min_kmh': matlab.double([30.0]),     # 默认最小速度
            'G2_s': matlab.double([2.0]),           # 默认时距参数
            'timestamp': matlab.double([time.time()]),

            # 新增的3个外部状态字段 (14/15-field版本)
            'external_stage_offset': matlab.double([self.sppvt_state.current_stage_offset]),
            'external_stage_manager_states': matlab.double([[0.0], [0.0], [0.0]]),  # 3x1列向量
            'external_adapter_states': matlab.double([[self.sppvt_state.prev_error],
                                                     [self.sppvt_state.prev_velocity],
                                                     [self.sppvt_state.prev_accel]])  # 3x1列向量
        }

        return simulink_inputs

    def run_single_step_simulation(self, control_error: float, ego_speed_ms: float,
                                 control_mode_flag: bool, control_enabled: bool) -> Optional[float]:
        """
        运行单步仿真

        Args:
            control_error: 控制误差
            ego_speed_ms: 自车速度 (m/s)
            control_mode_flag: 控制模式标志
            control_enabled: 控制使能

        Returns:
            SPPVT控制输出，失败时返回None
        """
        start_time = time.time()

        # 创建局部变量引用，避免IDE在深层嵌套中的未解析引用警告
        current_control_error = control_error
        current_ego_speed_ms = ego_speed_ms

        try:
            if not self.model_loaded:
                self._load_simulink_model()

            # 正确的总线数据创建方式 - 参考test_integrated_sppvt_model.m
            # 创建时间向量
            time_points = self.matlab_eng.eval(f"0:{self.dt/2}:{self.dt}")  # [0, dt/2, dt]
            ego_speed_kmh = ego_speed_ms * 3.6

            # 创建总线结构体，每个字段都是timeseries对象
            # DecisionSPPVTInputExtended (14字段)
            input_data_dict = {
                # 原有的11个字段 - double类型
                'ego_speed_kmh': self.matlab_eng.timeseries(
                    matlab.double([ego_speed_kmh, ego_speed_kmh, ego_speed_kmh]),
                    time_points, 'Name', 'ego_speed_kmh'
                ),
                'ego_speed_ms': self.matlab_eng.timeseries(
                    matlab.double([ego_speed_ms, ego_speed_ms, ego_speed_ms]),
                    time_points, 'Name', 'ego_speed_ms'
                ),
                'control_error': self.matlab_eng.timeseries(
                    matlab.double([control_error, control_error, control_error]),
                    time_points, 'Name', 'control_error'
                ),
                'V_target_kmh': self.matlab_eng.timeseries(
                    matlab.double([50.0, 50.0, 50.0]),
                    time_points, 'Name', 'V_target_kmh'
                ),
                'V_min_kmh': self.matlab_eng.timeseries(
                    matlab.double([30.0, 30.0, 30.0]),
                    time_points, 'Name', 'V_min_kmh'
                ),
                'G2_s': self.matlab_eng.timeseries(
                    matlab.double([2.0, 2.0, 2.0]),
                    time_points, 'Name', 'G2_s'
                ),
                'timestamp': self.matlab_eng.timeseries(
                    time_points, time_points, 'Name', 'timestamp'
                ),

                # int32类型字段
                'command_type': self.matlab_eng.timeseries(
                    matlab.int32([1, 1, 1]),
                    time_points, 'Name', 'command_type'
                ),
                'control_mode_flag': self.matlab_eng.timeseries(
                    matlab.int32([1 if control_mode_flag else 2,
                                 1 if control_mode_flag else 2,
                                 1 if control_mode_flag else 2]),
                    time_points, 'Name', 'control_mode_flag'
                ),

                # boolean类型字段
                'command_active': self.matlab_eng.timeseries(
                    matlab.logical([control_enabled, control_enabled, control_enabled]),
                    time_points, 'Name', 'command_active'
                ),
                'manual_throttle_active': self.matlab_eng.timeseries(
                    matlab.logical([False, False, False]),
                    time_points, 'Name', 'manual_throttle_active'
                ),

                # 新增的3个外部状态字段 - double类型
                'external_stage_offset': self.matlab_eng.timeseries(
                    matlab.double([self.sppvt_state.current_stage_offset,
                                  self.sppvt_state.current_stage_offset,
                                  self.sppvt_state.current_stage_offset]),
                    time_points, 'Name', 'external_stage_offset'
                ),
                'external_stage_manager_states': self.matlab_eng.timeseries(
                    matlab.double([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                    time_points, 'Name', 'external_stage_manager_states'
                ),
                'external_adapter_states': self.matlab_eng.timeseries(
                    matlab.double([[self.sppvt_state.prev_error, self.sppvt_state.prev_error, self.sppvt_state.prev_error],
                                  [self.sppvt_state.prev_velocity, self.sppvt_state.prev_velocity, self.sppvt_state.prev_velocity],
                                  [self.sppvt_state.prev_accel, self.sppvt_state.prev_accel, self.sppvt_state.prev_accel]]),
                    time_points, 'Name', 'external_adapter_states'
                )
            }

            # 为每个timeseries设置时间单位
            for field_name, ts_obj in input_data_dict.items():
                self.matlab_eng.workspace[f'{field_name}_temp'] = ts_obj
                self.matlab_eng.eval(f"{field_name}_temp.TimeInfo.Units = 'seconds';", nargout=0)
                input_data_dict[field_name] = self.matlab_eng.workspace[f'{field_name}_temp']

            # 将输入数据传入工作空间
            self.matlab_eng.workspace['input_data'] = input_data_dict

            # 配置外部输入为总线结构体
            self.matlab_eng.eval(f"set_param('{self.model_name}', 'LoadExternalInput', 'on')", nargout=0)
            self.matlab_eng.eval(f"set_param('{self.model_name}', 'ExternalInput', 'input_data')", nargout=0)

            # 运行仿真（使用eval方式，参考成功的实现）
            self.logger.debug(f"运行单步仿真: t={self.simulation_time:.3f}s")

            self.matlab_eng.eval(f"simOut = sim('{self.model_name}');", nargout=0)

            # 检查simOut是否存在
            if not self.matlab_eng.eval("exist('simOut', 'var')"):
                raise RuntimeError("仿真未能生成输出")

            # 更新状态并获取输出（使用eval方式）
            sppvt_output = self.update_state_from_simulink_output_eval(current_control_error, current_ego_speed_ms)

            if sppvt_output is not None:
                self.last_successful_output = sppvt_output
                self.error_count = 0  # 重置错误计数

                # 更新时间
                self.simulation_time += self.dt

                # 记录执行时间
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)

                self.logger.debug(f"仿真成功: 输出={sppvt_output:.3f}, 耗时={execution_time*1000:.1f}ms")

                return sppvt_output
            else:
                raise ValueError("无法获取有效的SPPVT输出")

        except Exception as e:
            self.error_count += 1
            execution_time = time.time() - start_time

            self.logger.error(f"单步仿真失败 (错误#{self.error_count}): {e}, 耗时={execution_time*1000:.1f}ms")

            # 错误恢复策略
            if self.error_count < self.config['max_consecutive_errors']:
                if self.last_successful_output is not None:
                    self.logger.warning(f"使用上次成功输出: {self.last_successful_output}")
                    return self.last_successful_output
            else:
                self.logger.critical("连续错误过多，需要重新初始化")
                self._reinitialize_system()

            return None

    def _reinitialize_system(self):
        """重新初始化系统"""
        try:
            self.logger.info("重新初始化SPPVT状态管理器...")

            # 重置状态
            self.sppvt_state = SPPVTState()
            self.simulation_time = 0.0
            self.error_count = 0

            # 重新加载模型
            self.model_loaded = False
            self._load_simulink_model()

            self.logger.info("系统重新初始化成功")

        except Exception as e:
            self.logger.error(f"系统重新初始化失败: {e}")
            raise

    def calculate_stage_offset_upgrade(self, current_accel: float, current_velocity: float,
                                     current_error: float) -> bool:
        """
        计算级差升级条件 (基于C代码分析的完整逻辑)

        升级条件: (acceleration < 0) && (|velocity| <= delta) && (|error| > eta)
        """
        delta = self.config['sppvt_delta']
        eta = self.config['sppvt_eta']

        condition1 = current_accel < 0  # 减速状态
        condition2 = abs(current_velocity) <= delta  # 速度足够小
        condition3 = abs(current_error) > eta  # 误差足够大

        should_upgrade = condition1 and condition2 and condition3

        if should_upgrade:
            # 计算新的级差值
            sign_error = 1.0 if current_error >= 0 else -1.0
            new_stage_offset = self.sppvt_state.current_stage_offset + sign_error * 0.1  # 级差步长

            self.logger.info(f"级差升级: {self.sppvt_state.current_stage_offset:.3f} -> {new_stage_offset:.3f}")
            self.sppvt_state.current_stage_offset = new_stage_offset

        return should_upgrade

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if not self.execution_times:
            return {"status": "no_data"}

        times_ms = [t * 1000 for t in self.execution_times]

        return {
            "avg_execution_time_ms": np.mean(times_ms),
            "max_execution_time_ms": np.max(times_ms),
            "min_execution_time_ms": np.min(times_ms),
            "std_execution_time_ms": np.std(times_ms),
            "total_simulations": len(times_ms),
            "error_count": self.error_count,
            "current_stage_offset": self.sppvt_state.current_stage_offset,
            "simulation_time": self.simulation_time
        }

    def reset_state(self):
        """重置SPPVT状态"""
        self.logger.info("重置SPPVT状态")
        self.sppvt_state = SPPVTState()
        self.simulation_time = 0.0
        self.error_count = 0
        self.execution_times.clear()

    def cleanup(self):
        """清理资源"""
        try:
            if self.matlab_eng and self.model_loaded:
                # 强制关闭模型，不保存更改
                self.matlab_eng.eval(f"close_system('{self.model_name}', 0)", nargout=0)
                self.logger.info("Simulink模型已关闭")
        except Exception as e:
            self.logger.warning(f"模型关闭时出现警告: {e}")
            # 尝试备选关闭方式
            try:
                self.matlab_eng.eval(f"bdclose('{self.model_name}')", nargout=0)
                self.logger.info("使用bdclose成功关闭模型")
            except:
                self.logger.warning("模型可能仍在MATLAB中打开，这是正常现象")

# 使用示例和测试函数
def test_realtime_sppvt_manager():
    """测试实时SPPVT状态管理器"""
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # 创建管理器
    manager = RealtimeSPPVTStateManager()

    try:
        # 模拟一系列控制步骤
        test_scenarios = [
            (0.5, 15.0, True, True),   # 正常跟车
            (1.2, 12.0, True, True),   # 大误差
            (-0.8, 8.0, True, True),   # 负误差，低速
            (0.1, 13.5, True, True),   # 小误差
        ]

        print("开始实时SPPVT测试...")

        for i, (error, speed, mode, enabled) in enumerate(test_scenarios):
            print(f"\n--- 测试步骤 {i+1} ---")
            print(f"输入: error={error}, speed={speed}, mode={mode}, enabled={enabled}")

            # 运行单步仿真
            output = manager.run_single_step_simulation(error, speed, mode, enabled)

            if output is not None:
                print(f"输出: sppvt_control = {output:.3f}")
            else:
                print("输出: 失败")

            # 显示状态
            state = manager.sppvt_state
            print(f"状态: stage_offset={state.current_stage_offset:.3f}, "
                  f"prev_error={state.prev_error:.3f}")

        # 显示性能统计
        stats = manager.get_performance_stats()
        print(f"\n性能统计: {stats}")

    finally:
        manager.cleanup()

if __name__ == "__main__":
    test_realtime_sppvt_manager()