"""
Real-time SPPVT State Manager
Solves state persistence issues when repeatedly calling sim() in Simulink
High cohesion, low coupling design specifically for managing all SPPVT-related states
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
    """SPPVT state data structure"""
    current_stage_offset: float = 0.0
    prev_error: float = 0.0
    prev_velocity: float = 13.89  # 50 km/h = 13.89 m/s
    prev_accel: float = 0.1
    last_update_time: float = 0.0

    # Historical records for analysis
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

class RealtimeSPPVTStateManager:
    """
    Real-time SPPVT State Manager

    Core functionality:
    1. Manage all persistent states of SPPVT algorithm
    2. Provide single-step simulation interface to avoid state reset
    3. Implement external state injection to ensure state continuity
    4. Performance monitoring and exception handling
    """

    def __init__(self, matlab_engine=None, model_name='ACC_Decision_SPPVT_Integrated'):
        self.logger = logging.getLogger(__name__)

        # MATLAB engine management
        self.matlab_eng = matlab_engine
        self.model_name = model_name
        self.model_loaded = False

        # State management
        self.sppvt_state = SPPVTState()
        self.simulation_time = 0.0
        self.dt = 0.02  # 20ms time step (updated from 50ms for faster response)

        # Performance monitoring
        self.execution_times = deque(maxlen=1000)
        self.error_count = 0
        self.last_successful_output = None

        # Configuration parameters
        self.config = {
            'sppvt_kp': 1.0,
            'max_accel': 2.0,
            'max_decel': -3.0,
            'sppvt_delta': 0.05,
            'sppvt_eta': 0.2,
            'timeout_seconds': 1.0,
            'max_consecutive_errors': 5
        }

        # Performance optimization - workspace templates and cached operations
        self.cached_time_points = None
        self.input_data_initialized = False
        self.workspace_templates_initialized = False

        self._initialize_matlab_engine()

    def _initialize_matlab_engine(self):
        """Initialize MATLAB engine"""
        try:
            if self.matlab_eng is None:
                self.logger.info("Starting MATLAB engine...")
                self.matlab_eng = matlab.engine.start_matlab()
                self.logger.info("MATLAB engine started successfully")

            # Load model
            self._load_simulink_model()

        except Exception as e:
            self.logger.error(f"MATLAB engine initialization failed: {e}")
            raise

    def _load_simulink_model(self):
        """Load Simulink model with Fast Restart optimization"""
        try:
            self.logger.info(f"Loading Simulink model: {self.model_name}")

            # First create required bus definitions
            self.logger.info("Creating 17/18-field bus definitions...")
            self.matlab_eng.eval("create_decision_sppvt_bus()", nargout=0)

            # Check if bus was successfully created
            bus_check = self.matlab_eng.eval("exist('DecisionSPPVTInputExtended', 'var')")
            if bus_check == 0:
                self.logger.error("DecisionSPPVTInputExtended bus creation failed")
                raise RuntimeError("Bus definition creation failed")

            self.logger.info("Success: 17/18-field bus definitions created")

            # Load model
            self.matlab_eng.eval(f"load_system('{self.model_name}')", nargout=0)

            # Configure simulation parameters with Fast Restart optimization
            # Fast Restart keeps model compiled between sim() calls - critical for performance!
            self.logger.info("Configuring Accelerator mode with Fast Restart...")
            self.matlab_eng.eval(f"""
                % Use accelerator mode for fast simulation
                set_param('{self.model_name}', 'SimulationMode', 'accelerator');

                % Enable Fast Restart - keeps model compiled between runs
                set_param('{self.model_name}', 'FastRestart', 'on');

                % Basic simulation parameters
                set_param('{self.model_name}', 'StopTime', '{self.dt}');
                set_param('{self.model_name}', 'SaveOutput', 'on');
                set_param('{self.model_name}', 'OutputSaveName', 'yout');
                set_param('{self.model_name}', 'SaveFormat', 'Dataset');

                % Disable model reference rebuild checks for performance
                set_param('{self.model_name}', 'UpdateModelReferenceTargets', 'AssumeUpToDate');

                disp('Fast Restart configuration complete');
            """, nargout=0)

            self.output_format = 'Dataset'
            self.logger.info("Configured Accelerator + Fast Restart mode (Dataset output)")

            # Perform initial compilation by running a dummy simulation
            # This triggers the accelerator build and prepares Fast Restart
            self.logger.info("Performing initial accelerator build (one-time ~10s)...")
            self._run_initial_compilation()

            self.model_loaded = True
            self.logger.info("✅ Simulink model ready with Fast Restart enabled")
            self.logger.info("   Subsequent simulations will use compiled model for maximum speed")

        except Exception as e:
            self.logger.error(f"Simulink model loading failed: {e}")
            # If bus-related error, provide detailed information
            if "Bus" in str(e) or "DecisionSPPVT" in str(e):
                self.logger.error("This might be a bus definition issue. Please check:")
                self.logger.error("1. create_decision_sppvt_bus.m file exists")
                self.logger.error("2. Simulink model uses correct bus names")
                self.logger.error("3. Inport/Outport configured for new 17/18-field buses")
            raise

    def _run_initial_compilation(self):
        """Run initial compilation to trigger accelerator build and prepare Fast Restart"""
        try:
            import time
            self.logger.info("Creating dummy input for initial compilation...")

            # Create minimal dummy input for compilation
            self.matlab_eng.eval(f"""
                % Create minimal dummy timeseries data
                time_pts = [0.0, {self.dt}];
                dummy_double = [0.0, 0.0];
                dummy_int = int32([1, 1]);
                dummy_bool = logical([true, true]);
                dummy_array = [[0.0; 0.0; 0.0], [0.0; 0.0; 0.0]];

                % Build complete input structure matching DecisionSPPVTInputExtended
                dummy_input.ego_speed_kmh = timeseries(dummy_double, time_pts, 'Name', 'ego_speed_kmh');
                dummy_input.ego_speed_ms = timeseries(dummy_double, time_pts, 'Name', 'ego_speed_ms');
                dummy_input.command_type = timeseries(dummy_int, time_pts, 'Name', 'command_type');
                dummy_input.command_active = timeseries(dummy_bool, time_pts, 'Name', 'command_active');
                dummy_input.manual_throttle_active = timeseries(dummy_bool, time_pts, 'Name', 'manual_throttle_active');
                dummy_input.control_error = timeseries(dummy_double, time_pts, 'Name', 'control_error');
                dummy_input.control_mode_flag = timeseries(dummy_int, time_pts, 'Name', 'control_mode_flag');
                dummy_input.V_target_kmh = timeseries([50.0, 50.0], time_pts, 'Name', 'V_target_kmh');
                dummy_input.V_min_kmh = timeseries([30.0, 30.0], time_pts, 'Name', 'V_min_kmh');
                dummy_input.G2_s = timeseries([2.0, 2.0], time_pts, 'Name', 'G2_s');
                dummy_input.timestamp = timeseries(time_pts, time_pts, 'Name', 'timestamp');
                dummy_input.external_stage_offset = timeseries(dummy_double, time_pts, 'Name', 'external_stage_offset');
                dummy_input.external_stage_manager_states = timeseries(dummy_array, time_pts, 'Name', 'external_stage_manager_states');
                dummy_input.external_adapter_states = timeseries(dummy_array, time_pts, 'Name', 'external_adapter_states');

                % Set time units
                fields = fieldnames(dummy_input);
                for i = 1:length(fields)
                    dummy_input.(fields{{i}}).TimeInfo.Units = 'seconds';
                end

                % Configure external input
                set_param('{self.model_name}', 'ExternalInput', 'dummy_input');
                set_param('{self.model_name}', 'LoadExternalInput', 'on');

                disp('Running initial compilation (this will take ~10 seconds)...');
            """, nargout=0)

            compile_start = time.time()

            # Run first simulation to trigger compilation
            self.matlab_eng.eval(f"simOut_initial = sim('{self.model_name}');", nargout=0)

            compile_time = time.time() - compile_start

            self.logger.info(f"Initial compilation completed in {compile_time:.1f}s")
            self.logger.info("Model is now compiled and Fast Restart is active")

        except Exception as e:
            self.logger.error(f"Initial compilation failed: {e}")
            raise

    def _initialize_workspace_templates(self):
        """Initialize workspace timeseries templates for performance optimization"""
        if self.workspace_templates_initialized:
            return

        try:
            self.logger.info("Initializing workspace timeseries templates...")

            # Create all timeseries templates in MATLAB workspace
            self.matlab_eng.eval(f"""
            % Pre-allocate time vector and default arrays
            time_points_template = [0.0, {self.dt}];
            double_template = [0.0, 0.0];
            int32_template = int32([1, 1]);
            logical_template = logical([true, true]);

            % Pre-create all timeseries templates with default values
            ts_ego_speed_kmh = timeseries(double_template, time_points_template, 'Name', 'ego_speed_kmh');
            ts_ego_speed_ms = timeseries(double_template, time_points_template, 'Name', 'ego_speed_ms');
            ts_control_error = timeseries(double_template, time_points_template, 'Name', 'control_error');
            ts_V_target_kmh = timeseries(double_template, time_points_template, 'Name', 'V_target_kmh');
            ts_V_min_kmh = timeseries(double_template, time_points_template, 'Name', 'V_min_kmh');
            ts_G2_s = timeseries(double_template, time_points_template, 'Name', 'G2_s');
            ts_timestamp = timeseries(time_points_template, time_points_template, 'Name', 'timestamp');

            % Int32 field templates
            ts_command_type = timeseries(int32_template, time_points_template, 'Name', 'command_type');
            ts_control_mode_flag = timeseries(int32_template, time_points_template, 'Name', 'control_mode_flag');

            % Logical field templates
            ts_command_active = timeseries(logical_template, time_points_template, 'Name', 'command_active');
            ts_manual_throttle_active = timeseries(logical_template, time_points_template, 'Name', 'manual_throttle_active');

            % External state field templates
            ts_external_stage_offset = timeseries(double_template, time_points_template, 'Name', 'external_stage_offset');
            ts_external_stage_manager_states = timeseries([[0.0, 0.0, 0.0]; [0.0, 0.0, 0.0]], time_points_template, 'Name', 'external_stage_manager_states');
            ts_external_adapter_states = timeseries([[0.0, 0.0, 0.0]; [0.0, 0.0, 0.0]], time_points_template, 'Name', 'external_adapter_states');

            % Set time units once for all templates (batch operation)
            template_names = {{'ts_ego_speed_kmh', 'ts_ego_speed_ms', 'ts_control_error', 'ts_V_target_kmh', ...
                             'ts_V_min_kmh', 'ts_G2_s', 'ts_timestamp', 'ts_command_type', 'ts_control_mode_flag', ...
                             'ts_command_active', 'ts_manual_throttle_active', 'ts_external_stage_offset', ...
                             'ts_external_stage_manager_states', 'ts_external_adapter_states'}};

            for i = 1:length(template_names)
                eval([template_names{{i}} '.TimeInfo.Units = ''seconds'';']);
            end

            disp('Workspace timeseries templates initialized successfully');
            """, nargout=0)

            self.workspace_templates_initialized = True
            self.logger.info("Workspace timeseries templates initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize workspace templates: {e}")
            self.workspace_templates_initialized = False
            raise

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

                                        # 🔒 范围检查：stage_offset必须在合理范围内 [-100, 100]
                                        if not np.isfinite(new_stage_offset) or abs(new_stage_offset) > 100:
                                            self.logger.warning(f"Invalid stage_offset detected: {new_stage_offset}, using safe value 0.0")
                                            new_stage_offset = 0.0

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
                    # Non-Dataset format - this should not happen with our configuration
                    # but provide fallback handling
                    self.logger.warning("Unexpected non-Dataset format encountered")
                    sppvt_output = 0.0

            except Exception as e:
                self.logger.error(f"Dataset access failed: {e}")
                # Get more detailed error information
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

                sppvt_output = 0.0  # Default value
                self.logger.warning("Could not extract SPPVT output, using default value")

            # Update timestamp and historical records
            self.sppvt_state.last_update_time = time.time()
            self.sppvt_state.error_history.append(self.sppvt_state.prev_error)
            self.sppvt_state.velocity_history.append(self.sppvt_state.prev_velocity)
            self.sppvt_state.accel_history.append(self.sppvt_state.prev_accel)

            return sppvt_output

        except Exception as e:
            self.logger.error(f"Failed to update state from Simulink output (eval): {e}")
            return None

    def _sanitize_value(self, value: float, default: float = 0.0) -> float:
        """清洗数据：将Inf/NaN替换为默认值"""
        if value is None or not np.isfinite(value):
            return default
        return value

    def run_single_step_simulation(self, control_error: float, ego_speed_ms: float,
                                 control_mode_flag: bool, control_enabled: bool) -> Optional[float]:
        """
        Run single-step simulation

        Args:
            control_error: Control error
            ego_speed_ms: Ego vehicle speed (m/s)
            control_mode_flag: Control mode flag
            control_enabled: Control enabled

        Returns:
            SPPVT control output, None if failed
        """
        start_time = time.time()

        # 清洗输入数据
        control_error = self._sanitize_value(control_error, 0.0)
        ego_speed_ms = self._sanitize_value(ego_speed_ms, 0.0)

        # 创建局部变量引用，避免IDE在深层嵌套中的未解析引用警告
        current_control_error = control_error
        current_ego_speed_ms = ego_speed_ms

        try:
            if not self.model_loaded:
                self._load_simulink_model()

            # STRATEGY A: Use workspace templates for maximum performance
            # Initialize workspace templates (only once)
            self._initialize_workspace_templates()

            # Use current Python data to update workspace templates (清洗所有状态值)
            ego_speed_kmh = self._sanitize_value(ego_speed_ms * 3.6, 0.0)
            current_stage_offset = self._sanitize_value(self.sppvt_state.current_stage_offset, 0.0)
            prev_error = self._sanitize_value(self.sppvt_state.prev_error, 0.0)
            prev_velocity = self._sanitize_value(self.sppvt_state.prev_velocity, 13.89)
            prev_accel = self._sanitize_value(self.sppvt_state.prev_accel, 0.1)

            # Update workspace timeseries with real-time Python data
            data_update_start = time.time()
            self.matlab_eng.eval(f"""
            % Update all timeseries data with current Python values
            ts_ego_speed_kmh.Data = [{ego_speed_kmh}, {ego_speed_kmh}];
            ts_ego_speed_ms.Data = [{ego_speed_ms}, {ego_speed_ms}];
            ts_control_error.Data = [{control_error}, {control_error}];
            ts_V_target_kmh.Data = [50.0, 50.0];
            ts_V_min_kmh.Data = [30.0, 30.0];
            ts_G2_s.Data = [2.0, 2.0];

            % Update int32 fields
            ts_command_type.Data = int32([0, 0]);
            ts_control_mode_flag.Data = int32([{1 if control_mode_flag else 2}, {1 if control_mode_flag else 2}]);

            % Update logical fields
            ts_command_active.Data = logical([{str(control_enabled).lower()}, {str(control_enabled).lower()}]);
            ts_manual_throttle_active.Data = logical([false, false]);

            % Update external state fields with current SPPVT state (已清洗)
            ts_external_stage_offset.Data = [{current_stage_offset}, {current_stage_offset}];
            ts_external_stage_manager_states.Data = [[0.0, 0.0, 0.0]; [0.0, 0.0, 0.0]];
            ts_external_adapter_states.Data = [[{prev_error}, {prev_velocity}, {prev_accel}];
                                              [{prev_error}, {prev_velocity}, {prev_accel}]];

            % Assemble input_data structure using updated templates
            input_data.ego_speed_kmh = ts_ego_speed_kmh;
            input_data.ego_speed_ms = ts_ego_speed_ms;
            input_data.control_error = ts_control_error;
            input_data.V_target_kmh = ts_V_target_kmh;
            input_data.V_min_kmh = ts_V_min_kmh;
            input_data.G2_s = ts_G2_s;
            input_data.timestamp = ts_timestamp;
            input_data.command_type = ts_command_type;
            input_data.control_mode_flag = ts_control_mode_flag;
            input_data.command_active = ts_command_active;
            input_data.manual_throttle_active = ts_manual_throttle_active;
            input_data.external_stage_offset = ts_external_stage_offset;
            input_data.external_stage_manager_states = ts_external_stage_manager_states;
            input_data.external_adapter_states = ts_external_adapter_states;
            """, nargout=0)
            data_update_time = time.time() - data_update_start

            # Configure external input and run simulation
            self.logger.debug(f"Running optimized simulation: t={self.simulation_time:.3f}s, data_update={data_update_time*1000:.1f}ms")

            # Configure and run simulation using SimulationInput for build control
            config_start = time.time()
            self.matlab_eng.eval(f"set_param('{self.model_name}', 'ExternalInput', 'input_data')", nargout=0)
            self.matlab_eng.eval(f"set_param('{self.model_name}', 'LoadExternalInput', 'on')", nargout=0)

            # 使用SimulationInput对象明确禁用build检查
            # 参考: https://www.mathworks.com/help/simulink/slref/simulink.simulationinput.html
            self.matlab_eng.eval(f"""
            simIn = Simulink.SimulationInput('{self.model_name}');
            simIn = simIn.setModelParameter('UpdateModelReferenceTargets', 'AssumeUpToDate');
            """, nargout=0)
            config_time = time.time() - config_start

            sim_start = time.time()
            # 使用SimulationInput运行仿真而不是直接sim()
            self.matlab_eng.eval(f"simOut = sim(simIn);", nargout=0)
            sim_time = time.time() - sim_start

            # 检查simOut是否存在
            if not self.matlab_eng.eval("exist('simOut', 'var')"):
                raise RuntimeError("Simulation failed to generate output")

            # Extract output with detailed timing
            extract_start = time.time()
            sppvt_output = self.update_state_from_simulink_output_eval(current_control_error, current_ego_speed_ms)
            extract_time = time.time() - extract_start

            if sppvt_output is not None:
                self.last_successful_output = sppvt_output
                self.error_count = 0  # 重置错误计数

                # 更新时间
                self.simulation_time += self.dt

                # 记录执行时间
                total_execution_time = time.time() - start_time
                self.execution_times.append(total_execution_time)

                # Detailed performance logging
                self.logger.debug(f"Strategy A Performance Breakdown:")
                self.logger.debug(f"  Data update: {data_update_time*1000:.1f}ms")
                self.logger.debug(f"  Config: {config_time*1000:.1f}ms")
                self.logger.debug(f"  Simulation: {sim_time*1000:.1f}ms")
                self.logger.debug(f"  Extraction: {extract_time*1000:.1f}ms")
                self.logger.debug(f"  Total: {total_execution_time*1000:.1f}ms")
                self.logger.debug(f"  SPPVT output: {sppvt_output:.3f}")

                return sppvt_output
            else:
                raise ValueError("无法获取有效的SPPVT输出")

        except Exception as e:
            self.error_count += 1
            execution_time = time.time() - start_time

            self.logger.error(f"Single-step simulation failed (error #{self.error_count}): {e}, time={execution_time*1000:.1f}ms")

            # Error recovery strategy
            if self.error_count < self.config['max_consecutive_errors']:
                if self.last_successful_output is not None:
                    self.logger.warning(f"Using last successful output: {self.last_successful_output}")
                    return self.last_successful_output
            else:
                self.logger.critical("Too many consecutive errors, system reinitialization required")
                self._reinitialize_system()

            return None

    def _reinitialize_system(self):
        """Reinitialize system"""
        try:
            self.logger.info("Reinitializing SPPVT state manager...")

            # Reset state
            self.sppvt_state = SPPVTState()
            self.simulation_time = 0.0
            self.error_count = 0

            # Reload model
            self.model_loaded = False
            self._load_simulink_model()

            self.logger.info("System reinitialization successful")

        except Exception as e:
            self.logger.error(f"System reinitialization failed: {e}")
            raise


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

