#!/usr/bin/env python3
"""
Simulink 决策 + 控制验证脚本
验证 ACC_Decision_SPPVT_Integrated.slx 的整体行为：
1. 状态转移逻辑 (S0-S3)
2. 决策输出正确性 (R1-R8)
3. SPPVT 控制输出
"""

import time
import numpy as np

from acc_decision_sppvt_interface import ACCDecisionSPPVTInterface
from acc_decision import ACCState


class SimulinkDecisionControlTester:
    """封装对集成模型的端到端验证"""

    def __init__(self, debug=True):
        self.debug = debug
        self.interface = ACCDecisionSPPVTInterface(debug=debug, use_realtime_sppvt=False)
        self.test_results = []
        self.failed_tests = []
        self.reset_sppvt_state()

    # ------------------------------------------------------------------
    # 状态外化工具
    # ------------------------------------------------------------------
    def reset_sppvt_state(self, stage_offset=0.0, stage_manager_states=None, adapter_states=None):
        self.stage_offset = float(stage_offset)
        self.stage_manager_states = list(stage_manager_states or [1.0, 0.0, 0.0])
        self.adapter_states = list(adapter_states or [0.0, 0.0, 0.0])

    def _to_state_list(self, value, fallback):
        if value is None:
            return list(fallback)
        if hasattr(value, "tolist"):
            return np.asarray(value).flatten().tolist()
        return list(value)

    def update_sppvt_state_from_result(self, result):
        if 'new_stage_offset' in result:
            try:
                self.stage_offset = float(np.asarray(result['new_stage_offset']).item())
            except (TypeError, ValueError):
                self.stage_offset = float(result['new_stage_offset'])

        if 'new_stage_manager_states' in result:
            self.stage_manager_states = self._to_state_list(
                result['new_stage_manager_states'],
                self.stage_manager_states
            )

        if 'new_adapter_states' in result:
            self.adapter_states = self._to_state_list(
                result['new_adapter_states'],
                self.adapter_states
            )

    def _build_adapter_state(self, ego_speed_kmh):
        adapter = (self.adapter_states + [0.0, 0.0, 0.0])[:3]
        if abs(adapter[1]) < 1e-6:
            adapter[1] = ego_speed_kmh / 3.6
        return adapter

    # ------------------------------------------------------------------
    # 输入构造 & 测试执行
    # ------------------------------------------------------------------
    def create_test_input(
        self,
        ego_speed_kmh,
        command_type,
        command_active=True,
        control_error=0.0,
        control_mode_flag=2,
        V_target_kmh=50.0,
        V_min_kmh=30.0,
        G2_s=2.0,
    ):
        return {
            'ego_speed_kmh': ego_speed_kmh,
            'ego_speed_ms': ego_speed_kmh / 3.6,
            'command_type': command_type,
            'command_active': command_active,
            'manual_throttle_active': False,
            'control_error': control_error,
            'control_mode_flag': control_mode_flag,
            'V_target_kmh': V_target_kmh,
            'V_min_kmh': V_min_kmh,
            'G2_s': G2_s,
            'timestamp': time.time(),
            'external_stage_offset': self.stage_offset,
            'external_stage_manager_states': list(self.stage_manager_states),
            'external_adapter_states': self._build_adapter_state(ego_speed_kmh),
        }

    def run_single_test(
        self,
        test_name,
        test_input,
        expected_state=None,
        expected_decision=None,
        check_control_enabled=None,
        description="",
        preserve_state=False,
    ):
        if not preserve_state:
            self.interface.reset()
            self.reset_sppvt_state()

        print(f"\n{'=' * 70}")
        print(f"[TEST] {test_name}")
        print(f"[DESC] {description}")
        print(f"{'=' * 70}")

        print("[INPUT] Input conditions:")
        print(f"   - 车速: {test_input['ego_speed_kmh']:.1f} km/h")
        print(f"   - 指令: I{test_input['command_type']-1 if test_input['command_type'] else 'NONE'} "
              f"(code={test_input['command_type']})")
        print(f"   - 激活: {test_input['command_active']}")
        print(f"   - 控制误差: {test_input['control_error']:.3f}")
        print(f"   - 控制模式: {'时距' if test_input['control_mode_flag'] == 1 else '速度'}")

        result = self.interface.process_decision_and_control(test_input)
        self.update_sppvt_state_from_result(result)

        print("\n[OUTPUT] Simulink output:")
        print(f"   - 状态: S{result['current_state']}")
        print(f"   - 决策: R{result['current_decision']}")
        print(f"   - 控制使能: {result['control_enabled']}")
        print(f"   - 扭矩仲裁: {result['torque_arbitration_active']}")
        print(f"   - 控制输出: {result['target_accel']:.3f} m/s²")

        success = True
        errors = []
        if expected_state is not None and result['current_state'] != expected_state:
            errors.append(f"状态错误：期望 S{expected_state}，实际 S{result['current_state']}")
        if expected_decision is not None and result['current_decision'] != expected_decision:
            errors.append(f"决策错误：期望 R{expected_decision}，实际 R{result['current_decision']}")
        if check_control_enabled is not None and result['control_enabled'] != check_control_enabled:
            errors.append(f"控制使能错误：期望 {check_control_enabled}，实际 {result['control_enabled']}")

        success = not errors
        print("\n[EXPECT] Expected result:")
        if expected_state is not None:
            print(f"   - 状态: S{expected_state}")
        if expected_decision is not None:
            print(f"   - 决策: R{expected_decision}")
        if check_control_enabled is not None:
            print(f"   - 控制使能: {check_control_enabled}")

        if success:
            print("\n[PASS] Test passed")
            self.test_results.append({'name': test_name, 'status': 'PASS', 'result': result})
        else:
            print("\n[FAIL] Test failed")
            for err in errors:
                print(f"   [WARN] {err}")
            self.test_results.append({'name': test_name, 'status': 'FAIL', 'errors': errors, 'result': result})
            self.failed_tests.append(test_name)

        return result

    # ------------------------------------------------------------------
    # 测试场景
    # ------------------------------------------------------------------
    def test_state_transitions(self):
        print(f"\n{'#' * 70}")
        print("# 第一部分: 状态转移逻辑测试")
        print(f"{'#' * 70}")

        # 1) S2 -> S0
        print(f"\n{'─' * 70}")
        print("测试组1: S2 -> S0 转移（无史启控）")
        print(f"{'─' * 70}")

        self.interface.reset()
        self.reset_sppvt_state()
        idle_input = self.create_test_input(ego_speed_kmh=45.0, command_type=0, command_active=False)
        result = self.interface.process_decision_and_control(idle_input)
        self.update_sppvt_state_from_result(result)
        print(f"[OK] Initialize to S{result['current_state']} state")

        self.run_single_test(
            "S2_I0_to_S0_R5",
            self.create_test_input(ego_speed_kmh=45.0, command_type=1, command_active=True),
            expected_state=0,
            expected_decision=5,
            check_control_enabled=True,
            description="S2 状态下按 I0 启控",
            preserve_state=True,
        )

        # 2) S0 -> S1
        self.run_single_test(
            "S0_I6_to_S1_R8",
            self.create_test_input(ego_speed_kmh=45.0, command_type=7, command_active=True),
            expected_state=1,
            expected_decision=8,
            check_control_enabled=False,
            description="Cancel command from S0 enters S1 standby",
            preserve_state=True,
        )

        # 3) S1 -> S0
        self.run_single_test(
            "S1_I1_to_S0_R6",
            self.create_test_input(ego_speed_kmh=45.0, command_type=2, command_active=True),
            expected_state=0,
            expected_decision=6,
            check_control_enabled=True,
            description="S1 resume control with I1 (inherit history)",
            preserve_state=True,
        )

        # 4) S0 内部行为
        # 4) S0 内部行为
        self.run_single_test(
            "S0_I0_to_S0_R1",
            self.create_test_input(ego_speed_kmh=50.0, command_type=1, command_active=True),
            expected_state=0,
            expected_decision=1,
            check_control_enabled=True,
            description="S0 adjust: I0 decreases target speed (R1)",
            preserve_state=True,
        )

        self.run_single_test(
            "S0_I1_to_S0_R2",
            self.create_test_input(ego_speed_kmh=45.0, command_type=2, command_active=True),
            expected_state=0,
            expected_decision=2,
            check_control_enabled=True,
            description="S0 adjust: I1 increases target speed (R2)",
            preserve_state=True,
        )

        self.run_single_test(
            "S0_I2_to_S0_R3",
            self.create_test_input(ego_speed_kmh=45.0, command_type=3, command_active=True),
            expected_state=0,
            expected_decision=3,
            check_control_enabled=True,
            description="S0 adjust: I2 decreases time gap (R3)",
            preserve_state=True,
        )

        self.run_single_test(
            "S0_I3_to_S0_R4",
            self.create_test_input(ego_speed_kmh=45.0, command_type=4, command_active=True),
            expected_state=0,
            expected_decision=4,
            check_control_enabled=True,
            description="S0 adjust: I3 increases time gap (R4)",
            preserve_state=True,
        )

        # 5) S3 low-speed state
        print()
        print("-" * 70)
        print("State group 5: S3 low-speed state")
        print("-" * 70)

        self.run_single_test(
            "Low_Speed_S3",
            self.create_test_input(ego_speed_kmh=25.0, command_type=0, command_active=False, V_min_kmh=30.0),
            expected_state=3,
            expected_decision=8,
            check_control_enabled=False,
            description="Below V_min enters S3 low-speed standby",
            preserve_state=False,
        )

        self.run_single_test(
            "S3_I0_stays_S3",
            self.create_test_input(ego_speed_kmh=25.0, command_type=1, command_active=True, V_min_kmh=30.0),
            expected_state=3,
            expected_decision=8,
            check_control_enabled=False,
            description="Commands ignored in S3 low-speed standby",
            preserve_state=True,
        )
    def test_decision_outputs(self):
        print(f"\n{'#' * 70}")
        print("# 第二部分: 决策输出验证 (R1-R8)")
        print(f"{'#' * 70}")

        def prepare_active_state():
            self.interface.reset()
            self.reset_sppvt_state()
            init_input = self.create_test_input(ego_speed_kmh=45.0, command_type=1, command_active=True)
            result = self.interface.process_decision_and_control(init_input)
            self.update_sppvt_state_from_result(result)

        # R1-R4（在控状态）
        adjustments = [
            ('R1_Speed_Decrease', 1, 1, 'R1: decrease target speed (I0)'),
            ('R2_Speed_Increase', 2, 2, 'R2: increase target speed (I1)'),
            ('R3_TimeGap_Decrease', 3, 3, 'R3: decrease time gap (I2)'),
            ('R4_TimeGap_Increase', 4, 4, 'R4: increase time gap (I3)'),
        ]
        for test_name, cmd, expected_decision, desc in adjustments:
            prepare_active_state()
            self.run_single_test(
                test_name,
                self.create_test_input(ego_speed_kmh=45.0, command_type=cmd, command_active=True),
                expected_state=0,
                expected_decision=expected_decision,
                check_control_enabled=True,
                description=desc,
                preserve_state=True,
            )

        # R5 no-history control (S2 state)
        self.interface.reset()
        self.reset_sppvt_state()
        self.run_single_test(
            "R5_NoHistory_Control",
            self.create_test_input(ego_speed_kmh=45.0, command_type=1, command_active=True),
            expected_state=0,
            expected_decision=5,
            check_control_enabled=True,
            description="S2 state I0 command -> R5",
            preserve_state=False,
        )

        # R6 inherited control (S1 state)
        prepare_active_state()
        self.run_single_test(
            "R6_History_Control",
            self.create_test_input(ego_speed_kmh=45.0, command_type=7, command_active=True),
            expected_state=1,
            expected_decision=8,
            check_control_enabled=False,
            description="Enter S1 standby via cancel command",
            preserve_state=True,
        )
        self.run_single_test(
            "R6_History_Control_Rejoin",
            self.create_test_input(ego_speed_kmh=45.0, command_type=2, command_active=True),
            expected_state=0,
            expected_decision=6,
            check_control_enabled=True,
            description="S1 resume control using I1 -> R6",
            preserve_state=True,
        )

        # R7 torque arbitration
        prepare_active_state()
        self.run_single_test(
            "R7_Torque_Arbitration",
            self.create_test_input(ego_speed_kmh=50.0, command_type=5, command_active=True),
            expected_state=0,
            expected_decision=7,
            check_control_enabled=True,
            description="R7 扭矩仲裁触发",
            preserve_state=True,
        )

        # R8 系统待命（低速 & 取消）
        self.run_single_test(
            "R8_System_Standby",
            self.create_test_input(ego_speed_kmh=20.0, command_type=0, command_active=False, V_min_kmh=30.0),
            expected_state=3,
            expected_decision=8,
            check_control_enabled=False,
            description="低速待命 -> R8",
            preserve_state=False,
        )

    def test_sppvt_control(self):
        print(f"\n{'#' * 70}")
        print("# 第三部分: SPPVT 控制输出测试")
        print(f"{'#' * 70}")

        # 1) 时距控制
        self.interface.reset()
        self.reset_sppvt_state()
        result_time = self.run_single_test(
            "SPPVT_Time_Control",
            self.create_test_input(
                ego_speed_kmh=45.0,
                command_type=0,
                command_active=False,
                control_error=1.5,
                control_mode_flag=1,
            ),
            expected_state=None,
            expected_decision=None,
            check_control_enabled=False,
            description="时距误差 1.5 s -> 控制输出合理",
            preserve_state=False,
        )
        if not (-4.0 <= result_time['target_accel'] <= 3.0):
            self.failed_tests.append('SPPVT_Time_Control_Limits')
            print("[WARN] Time control output out of safe range [-4, 3] m/s^2")

        # 2) 速度控制
        self.interface.reset()
        self.reset_sppvt_state()
        result = self.run_single_test(
            "SPPVT_Speed_Control",
            self.create_test_input(
                ego_speed_kmh=45.0,
                command_type=0,
                command_active=False,
                control_error=2.5,
                control_mode_flag=2,
            ),
            expected_state=None,
            expected_decision=None,
            check_control_enabled=False,
            description="速度误差 2.5 m/s -> 控制输出合理",
            preserve_state=False,
        )
        if not (-4.0 <= result['target_accel'] <= 3.0):
            self.failed_tests.append('SPPVT_Speed_Control_Limits')
            print("[WARN] Speed control output out of safe range [-4, 3] m/s^2")

        # 3) 升级机制
        self.interface.reset()
        self.reset_sppvt_state()
        print(f"\n{'─' * 70}")
        print("测试组: SPPVT 阶段升级机制")
        print(f"{'─' * 70}")

        stage_offsets = []
        for _ in range(5):
            r = self.run_single_test(
                "SPPVT_Upgrade_Step",
                self.create_test_input(
                    ego_speed_kmh=45.0,
                    command_type=0,
                    command_active=False,
                    control_error=0.5,
                    control_mode_flag=1,
                ),
                expected_state=None,
                expected_decision=None,
                check_control_enabled=False,
                description="持续中等误差观察级差变化",
                preserve_state=True,
            )
            stage_offsets.append(r.get('sppvt_stage_output', 0.0))
            time.sleep(0.05)

        if len(stage_offsets) >= 2 and stage_offsets[-1] == stage_offsets[0]:
            self.failed_tests.append('SPPVT_Upgrade_NoChange')
            print("[WARN] Stage offset unchanged, upgrade logic may be ineffective")

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    def print_summary(self, elapsed_time):
        print(f"\n{'=' * 70}")
        print("[SUMMARY] Test Summary")
        print(f"{'=' * 70}")

        total = len(self.test_results)
        passed = sum(1 for t in self.test_results if t['status'] == 'PASS')
        failed = total - passed

        print(f"Total tests: {total}")
        print(f"[PASS] Passed: {passed}")
        print(f"[FAIL] Failed: {failed}")
        if self.failed_tests:
            print("Failed tests:")
            for name in self.failed_tests:
                print(f"   - {name}")
        else:
            print("[SUCCESS] All tests passed!")
        print(f"[TIME] Total time: {elapsed_time:.2f}s")
        print(f"{'=' * 70}")

    def run_all_tests(self):
        print(f"\n{'#' * 70}")
        print("# Simulink 决策 + 控制综合测试")
        print(f"# 目标模型: ACC_Decision_SPPVT_Integrated.slx")
        print(f"{'#' * 70}")

        start = time.time()
        self.test_state_transitions()
        self.test_decision_outputs()
        self.test_sppvt_control()
        self.print_summary(time.time() - start)


if __name__ == "__main__":
    print(f"\n{'=' * 70}")
    print("启动 Simulink 决策 + 控制测试框架")
    print(f"{'=' * 70}")
    tester = SimulinkDecisionControlTester(debug=False)
    tester.run_all_tests()
