"""
系统状态对象定义
用于封装分散的状态变量，提高代码可读性和可维护性
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VehicleState:
    """车辆状态"""
    speed_kmh: float = 0.0
    speed_ms: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0


@dataclass
class IncrementalControlState:
    """增量控制状态"""
    Y_prev: float = 0.0           # Y(k-1): 上一时刻输出扭矩
    e_i_prev: float = 0.0         # e_i(k-1): 上一时刻增强误差
    is_initialized: bool = False  # 是否已初始化

    def reset(self):
        """复位状态"""
        self.Y_prev = 0.0
        self.e_i_prev = 0.0
        self.is_initialized = False


@dataclass
class ACCState:
    """ACC系统状态"""
    system_enabled: bool = False  # ACC主开关
    control_enabled: bool = False  # ACC控制激活
    target_speed_kmh: float = 50.0
    time_gap_s: float = 2.0
    min_speed_kmh: float = 30.0
    torque_arbitration_active: bool = False
    current_decision: str = "STANDBY"
    current_control_mode: str = "NONE"
    incremental_control: IncrementalControlState = field(default_factory=IncrementalControlState)
    incremental_torque_nm: float = 0.0


@dataclass
class ManualInputState:
    """手动输入状态"""
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0

    # 按键状态
    w_pressed: bool = False
    s_pressed: bool = False
    a_pressed: bool = False
    d_pressed: bool = False

    def is_any_key_pressed(self) -> bool:
        """检查是否有任何按键被按下"""
        return self.w_pressed or self.s_pressed or self.a_pressed or self.d_pressed

    def has_throttle_input(self) -> bool:
        """检查是否有油门输入"""
        return self.w_pressed or self.throttle > 0.0

    def has_brake_input(self) -> bool:
        """检查是否有刹车输入"""
        return self.s_pressed or self.brake > 0.0


@dataclass
class PerceptionData:
    """感知数据"""
    vehicle_distance: float = 0.0
    lane_offset: float = 0.0
    has_target: bool = False
    target_speed_kmh: float = 0.0
    desired_distance: float = 0.0
    distance_error: float = 0.0
    control_error: float = 0.0
    control_mode_flag: int = 0  # 1=TIME, 2=SPEED
    control_mode_name: str = "Unknown"


@dataclass
class KeyboardCommand:
    """键盘指令"""
    code: int = 0  # 0=NONE, 1=I0降速, 2=I1增速, 3=I2降距, 4=I3增距, 5=I4油门, 6=I5刹车, 7=I6取消
    description: str = "NONE"

    def is_valid(self) -> bool:
        """检查指令是否有效"""
        return self.code > 0


@dataclass
class SystemState:
    """系统总状态（组合模式）"""
    ego: VehicleState = field(default_factory=VehicleState)
    target: VehicleState = field(default_factory=VehicleState)
    acc: ACCState = field(default_factory=ACCState)
    manual: ManualInputState = field(default_factory=ManualInputState)
    perception: PerceptionData = field(default_factory=PerceptionData)

    # 仿真时间
    sim_elapsed_s: float = 0.0
    frame_count: int = 0

    # 待处理的键盘指令
    pending_command: Optional[KeyboardCommand] = None

    def reset_pending_command(self):
        """重置待处理的指令"""
        self.pending_command = None

    def set_pending_command(self, code: int, description: str):
        """设置待处理的指令"""
        self.pending_command = KeyboardCommand(code=code, description=description)


@dataclass
class ControlOutput:
    """控制输出"""
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    mode: str = "MANUAL"
    torque_arbitration: bool = False
    sppvt_throttle: float = 0.0
    driver_throttle: float = 0.0

    def to_dict(self):
        """转换为字典格式（兼容旧代码）"""
        return {
            'throttle': self.throttle,
            'brake': self.brake,
            'steer': self.steer,
            'mode': self.mode,
            'torque_arbitration': self.torque_arbitration,
            'sppvt_throttle': self.sppvt_throttle,
            'driver_throttle': self.driver_throttle
        }


@dataclass
class StepResult:
    """单步执行结果"""
    control_output: ControlOutput
    system_state: SystemState
    unified_output: dict  # Simulink输出
    env_data: dict  # 环境数据
    simulink_duration_ms: float = 0.0

    def __post_init__(self):
        """初始化后处理"""
        # 确保env_data包含所有必要字段
        if not self.env_data:
            self.env_data = {}
