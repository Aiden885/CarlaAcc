"""
增强横向控制器（PID + 预瞄）
整合当前偏移和前瞻偏移两种反馈
"""

from lateral_pid_controller import LateralPIDController


class EnhancedLateralController:
    """
    增强横向控制器

    组合两种误差：
    1. 当前横向偏移（即时反应）
    2. 前瞻横向偏移（预判弯道）
    """

    def __init__(self, kp=0.02, ki=0.02, kd=0.4):
        """
        初始化增强横向控制器

        Args:
            kp, ki, kd: PID参数
        """
        # 底层PID控制器
        self.pid = LateralPIDController(kp=kp, ki=ki, kd=kd)

        # === 预瞄控制参数（可调） ===
        # 🔧 调整这些参数来改变横向控制特性
        self.weight_current = 0.7       # α: 当前偏移权重（主要反馈）
        self.weight_lookahead = 0.3     # β: 前瞻偏移权重（预判能力）

        # 预瞄距离配置
        self.lookahead_distance_base = 8.0    # 基础预瞄距离（米）
        self.lookahead_gain = 0.3             # 速度增益系数（秒）
        self.lookahead_min = 5.0              # 最小预瞄距离（米）
        self.lookahead_max = 15.0             # 最大预瞄距离（米）

    def calculate_lookahead_distance(self, speed_ms):
        """
        根据车速动态计算预瞄距离

        Args:
            speed_ms: 车速（m/s）

        Returns:
            float: 预瞄距离（米）
        """
        # L = L_base + gain * v
        lookahead = self.lookahead_distance_base + self.lookahead_gain * speed_ms

        # 限制在合理范围内
        lookahead = max(self.lookahead_min, min(self.lookahead_max, lookahead))

        return lookahead

    def update(self, current_offset, lookahead_offset, speed_ms, dt=0.05):
        """
        更新增强横向控制器

        Args:
            current_offset: 当前横向偏移（米，左负右正）
            lookahead_offset: 前瞻横向偏移（米，左负右正）
            speed_ms: 当前车速（m/s）
            dt: 时间步长（秒）

        Returns:
            float: 转向控制输出（-1到1）
        """
        # 组合误差：total_error = α*current_error + β*lookahead_error
        combined_error = (
            self.weight_current * current_offset +
            self.weight_lookahead * lookahead_offset
        )

        # PID计算转向输出
        steer_output = self.pid.update(combined_error, dt=dt)

        return steer_output

    def reset(self):
        """重置控制器状态"""
        self.pid.reset()

    def set_weights(self, weight_current=None, weight_lookahead=None):
        """
        动态调整权重系数

        Args:
            weight_current: 当前偏移权重
            weight_lookahead: 前瞻偏移权重
        """
        if weight_current is not None:
            self.weight_current = weight_current
        if weight_lookahead is not None:
            self.weight_lookahead = weight_lookahead

        # 归一化权重（确保总和为1.0）
        total = self.weight_current + self.weight_lookahead
        if total > 0:
            self.weight_current /= total
            self.weight_lookahead /= total

    def set_lookahead_params(self, base=None, gain=None, min_dist=None, max_dist=None):
        """
        动态调整预瞄距离参数

        Args:
            base: 基础预瞄距离（米）
            gain: 速度增益系数（秒）
            min_dist: 最小预瞄距离（米）
            max_dist: 最大预瞄距离（米）
        """
        if base is not None:
            self.lookahead_distance_base = base
        if gain is not None:
            self.lookahead_gain = gain
        if min_dist is not None:
            self.lookahead_min = min_dist
        if max_dist is not None:
            self.lookahead_max = max_dist

    def get_state(self):
        """
        获取控制器状态（用于调试）

        Returns:
            dict: 控制器状态信息
        """
        return {
            'weight_current': self.weight_current,
            'weight_lookahead': self.weight_lookahead,
            'lookahead_base': self.lookahead_distance_base,
            'lookahead_gain': self.lookahead_gain,
            'pid_state': self.pid.get_state()
        }
