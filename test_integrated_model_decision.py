#!/usr/bin/env python3
"""
End-to-end test for simplified acc_integrated_model.

Input vector (5 elements):
  [command_type, ego_speed_ms, vehicle_distance,
   target_speed_ms, current_engine_torque]

Output vector (2 elements):
  [control_enabled, final_output]

Simulink internal state (not directly observable):
  - current_state, has_history, last_active_decision (Unit Delay self-loops)
  - G2_s (G2_Manager, initial=4.0)
  - reset_flag (Reset_Flag_Detector)
  - Y0 (Y0_Latch)
  - Low_Speed_Detection (ego_speed < V_min → force S3/S2)
  - Torque_Arbitration (decision==7 → max(control_output, engine_torque))

Prerequisites:
  1) In MATLAB: run start_integrated_simulink_server.m
  2) Ensure decision_lookup_data.mat is loaded
  3) Ensure all subsystems are wired in the model
"""
import time
from simulink_udp_interface import SimulinkUDPClient


G2_INIT = 4.0   # G2_Manager initial value (Unit Delay IC)
TGT_SPD = 12.0  # target_speed_ms (lead vehicle speed, display only)


def make_input(cmd, speed, dist, tgt_spd, engine_torque):
    """构造5元素输入向量"""
    return [float(x) for x in [cmd, speed, dist, tgt_spd, engine_torque]]


def compute_expected_error(speed, dist, g2):
    """
    计算 Error_Calculation 子系统应输出的误差值（时距模式）
    error = dist / max(speed, 0.1) - g2
    """
    safe_speed = max(speed, 0.1)
    actual_gap = dist / safe_speed
    return actual_gap - g2


def measure_with_fresh_sppvt(client, y0, speed, dist, tgt_spd):
    """
    Cancel → reactivate → 读取首帧输出。

    通过 cancel 触发 control_enabled 下降沿 → Reset_Flag_Detector 产生 reset_flag
    → SPPVT 的 stage_offset 归零。再 activate 时 Y0_Latch 锁存 engine_torque。
    首帧输出 ≈ Kp * error + Y0（stage_offset ≈ 0），可以隔离误差对输出的纯影响。
    """
    # Cancel → S1, 触发 reset (control_enabled 下降沿)
    client.call(make_input(7, speed, dist, tgt_spd, 0.0))
    time.sleep(0.02)
    # Activate → S0, 锁存 Y0, SPPVT 从 stage_offset=0 开始
    client.call(make_input(1, speed, dist, tgt_spd, y0))
    time.sleep(0.02)
    # 首帧 idle: output ≈ Kp * error + Y0
    out = client.call(make_input(0, speed, dist, tgt_spd, 99.0))
    time.sleep(0.02)
    return out[1]  # final_output


def run_all_tests():
    client = SimulinkUDPClient(
        send_port=27000,
        recv_port=27001,
        num_inputs=5,
        num_outputs=2,
        timeout=2.0,
        debug=False,
        local_send_port=9090,
        send_initial_packet=False,
    )

    all_pass = True

    # ================================================================
    # Phase 1: Basic Connectivity
    # ================================================================
    print("=" * 70)
    print("Phase 1: Basic Connectivity (5 inputs -> 2 outputs)")
    print("=" * 70)

    # Internal state starts at S2 (Unit Delay IC=2), cmd=0 → stays S2
    test_in = make_input(0, 15.0, 30.0, TGT_SPD, 10.0)
    try:
        out = client.call(test_in)
        if len(out) == 2 and all(abs(v) < 1e6 for v in out):
            enabled = bool(int(round(out[0])))
            print(f"  [PASS] Got 2 outputs: control_enabled={enabled}, "
                  f"final_output={out[1]:.4f}")
        else:
            print(f"  [FAIL] Unexpected output: {out}")
            all_pass = False
    except Exception as e:
        print(f"  [FAIL] Connection error: {e}")
        client.cleanup()
        return

    time.sleep(0.02)

    # ================================================================
    # Phase 2: State Transitions via Commands
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 2: State Transitions (observe control_enabled)")
    print("=" * 70)

    # Internal state is S2 after Phase 1 (S2 + cmd=0 → stays S2)
    # We verify transitions by observing control_enabled changes

    cases = [
        # (description, command, expected_control_enabled)
        ("S2 + cmd=1 (E key) -> S0 activate",   1, True),
        ("S0 + cmd=0 (idle) -> stays S0",        0, True),
        ("S0 + cmd=7 (cancel) -> S1",            7, False),
        ("S1 + cmd=0 (idle) -> stays S1",        0, False),
        ("S1 + cmd=1 (E key) -> S0 activate",   1, True),
        ("S0 + cmd=6 (S key) -> S1 exit",        6, False),
        ("S1 + cmd=2 (Q key) -> S0 inherit",    2, True),
        ("S0 + cmd=7 (cancel) -> S1",            7, False),
    ]

    passed = 0
    failed = 0
    for name, cmd, expected_enabled in cases:
        inp = make_input(cmd, 15.0, 30.0, TGT_SPD, 10.0)
        out = client.call(inp)
        actual_enabled = bool(int(round(out[0])))
        ok = actual_enabled == expected_enabled
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: "
              f"control_enabled={actual_enabled} (expected {expected_enabled})")
        if ok:
            passed += 1
        else:
            failed += 1
            all_pass = False
        time.sleep(0.02)

    print(f"  Summary: {passed} passed, {failed} failed")

    # ================================================================
    # Phase 3: Error Calculation + Y0_Latch Verification
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 3: Error Calculation + Y0_Latch Verification")
    print("=" * 70)

    # After Phase 2, internal state is S1 (last case was cancel -> S1)

    # --- 3a: Latch Y0 and extract Kp ---
    print("\n  --- 3a: Latch Y0 and extract Kp ---")

    y0_test = 10.0
    speed_test = 10.0
    # G2 is internal, initially 4.0 (may have changed if T/R keys were sent)
    # Phase 2 only used cmd=0,1,2,6,7, so G2 should still be 4.0

    # Activate from S1: cmd=1 -> S0, control_enabled rising edge -> latch Y0
    inp_activate = make_input(1, speed_test, 40.0, TGT_SPD, y0_test)
    out_activate = client.call(inp_activate)
    time.sleep(0.02)
    print(f"    Activated: control_enabled={bool(int(round(out_activate[0])))}, "
          f"latched Y0={y0_test}")

    # Now in S0. Error_Calculation uses internal G2=4.0
    # Case 1: dist=40 -> gap=40/10=4.0, error=4.0-4.0=0.0
    # Case 2: dist=50 -> gap=50/10=5.0, error=5.0-4.0=1.0
    inp1 = make_input(0, speed_test, 40.0, TGT_SPD, 99.0)
    inp2 = make_input(0, speed_test, 50.0, TGT_SPD, 99.0)

    out1 = client.call(inp1)
    time.sleep(0.02)
    out2 = client.call(inp2)
    time.sleep(0.02)

    y1 = out1[1]  # final_output
    y2 = out2[1]
    e1 = compute_expected_error(speed_test, 40.0, G2_INIT)  # 0.0
    e2 = compute_expected_error(speed_test, 50.0, G2_INIT)  # 1.0

    print(f"    Case 1: dist=40, expected_error={e1:.2f}, Y={y1:.4f}")
    print(f"    Case 2: dist=50, expected_error={e2:.2f}, Y={y2:.4f}")

    kp = None
    if abs(e2 - e1) < 1e-9:
        print("    [FAIL] Cannot extract Kp: errors are identical")
        all_pass = False
    else:
        kp = (y2 - y1) / (e2 - e1)
        y0_actual = y1 - kp * e1
        print(f"    Extracted: Kp={kp:.4f}, "
              f"Y0_from_output={y0_actual:.4f} (latched Y0={y0_test})")
        if abs(y0_actual - y0_test) < 1.0:
            print(f"    [PASS] Y0 match (diff={abs(y0_actual - y0_test):.4f})")
        else:
            print(f"    [WARN] Y0 mismatch: expected ~{y0_test}, "
                  f"got {y0_actual:.4f}")

    # --- 3b: Linearity (varying distance) ---
    print("\n  --- 3b: Linearity (varying distance) ---")
    if kp is not None:
        # Note: SPPVT has evolving stage_offset, so exact linearity may drift
        test_dists = [30.0, 35.0, 45.0, 55.0, 60.0]
        verify_passed = 0
        for dist in test_dists:
            inp = make_input(0, speed_test, dist, TGT_SPD, 99.0)
            out = client.call(inp)
            time.sleep(0.02)
            y_actual = out[1]
            e_expected = compute_expected_error(speed_test, dist, G2_INIT)
            print(f"    dist={dist:5.1f}: error={e_expected:6.2f}, "
                  f"Y_actual={y_actual:8.4f}")
            verify_passed += 1
        print(f"    Linearity: {verify_passed}/{len(test_dists)} printed "
              f"(visual check for monotonic trend)")

    # --- 3c: Varying speed (same distance) ---
    print("\n  --- 3c: Varying speed (dist=60, G2=4.0) ---")
    for spd in [10.0, 12.0, 15.0, 18.0, 20.0]:
        inp = make_input(0, spd, 60.0, TGT_SPD, 99.0)
        out = client.call(inp)
        time.sleep(0.02)
        y_actual = out[1]
        e_expected = compute_expected_error(spd, 60.0, G2_INIT)
        print(f"    speed={spd:5.1f}: gap={60.0/max(spd,0.1):.2f}s, "
              f"error={e_expected:6.2f}, Y_actual={y_actual:8.4f}")

    # --- 3d: target_speed_ms does NOT affect control output ---
    print("\n  --- 3d: target_speed_ms does NOT affect control output ---")
    inp_tgt0 = make_input(0, 10.0, 40.0, 0.0, 99.0)
    inp_tgt20 = make_input(0, 10.0, 40.0, 20.0, 99.0)
    out_tgt0 = client.call(inp_tgt0)
    time.sleep(0.02)
    out_tgt20 = client.call(inp_tgt20)
    time.sleep(0.02)
    diff_tgt = abs(out_tgt0[1] - out_tgt20[1])
    if diff_tgt < 0.01:
        print(f"    [PASS] target_speed=0 vs 20: Y diff={diff_tgt:.6f} (no effect)")
    else:
        print(f"    [FAIL] target_speed affects output: diff={diff_tgt:.4f}")
        all_pass = False

    # --- 3e: Y0_Latch verification ---
    print("\n  --- 3e: Y0_Latch (different Y0 values) ---")

    # Cancel to S1
    client.call(make_input(7, 15.0, 30.0, TGT_SPD, 0.0))
    time.sleep(0.02)

    # Activate with Y0=10 (rising edge latches engine_torque=10)
    client.call(make_input(1, 10.0, 40.0, TGT_SPD, 10.0))
    time.sleep(0.02)
    # Read output at zero error (dist=40, speed=10, G2=4 -> error=0)
    out_ref_y10 = client.call(make_input(0, 10.0, 40.0, TGT_SPD, 999.0))
    time.sleep(0.02)
    print(f"    Latched Y0=10: final_output={out_ref_y10[1]:.4f}")

    # Cancel and reactivate with Y0=20
    client.call(make_input(7, 15.0, 30.0, TGT_SPD, 0.0))
    time.sleep(0.02)
    client.call(make_input(1, 10.0, 40.0, TGT_SPD, 20.0))
    time.sleep(0.02)
    out_ref_y20 = client.call(make_input(0, 10.0, 40.0, TGT_SPD, 999.0))
    time.sleep(0.02)
    print(f"    Latched Y0=20: final_output={out_ref_y20[1]:.4f}")

    y_shift = out_ref_y20[1] - out_ref_y10[1]
    print(f"    Shift={y_shift:.4f} (expected ~10.0)")
    if abs(y_shift - 10.0) < 2.0:
        print(f"    [PASS] Y0_Latch shift correct")
    else:
        print(f"    [FAIL] Y0_Latch shift incorrect "
              f"(expected 10.0, got {y_shift:.4f})")
        all_pass = False

    # --- 3f: Y0_Latch holds after activation ---
    print("\n  --- 3f: Y0_Latch holds (input change after latch has no effect) ---")
    # Currently in S0 with Y0=20 latched
    inp_hold1 = make_input(0, 10.0, 40.0, TGT_SPD, 20.0)
    inp_hold2 = make_input(0, 10.0, 40.0, TGT_SPD, 50.0)
    out_hold1 = client.call(inp_hold1)
    time.sleep(0.02)
    out_hold2 = client.call(inp_hold2)
    time.sleep(0.02)
    diff_hold = abs(out_hold1[1] - out_hold2[1])
    if diff_hold < 0.5:
        print(f"    [PASS] engine_torque 20 vs 50 while latched: "
              f"Y diff={diff_hold:.6f}")
    else:
        print(f"    [FAIL] Y0_Latch did not hold: diff={diff_hold:.4f}")
        all_pass = False

    # ================================================================
    # Phase 4: G2_Manager Test
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 4: G2_Manager (T/R key adjusts G2)")
    print("=" * 70)
    print("  Strategy: cancel->activate between each measurement to reset SPPVT,")
    print("  so output = Kp * error + Y0 (stage_offset=0), isolating G2 effect.\n")

    # 按文档 decision_types_explanation.md：
    #   T键(cmd=3) = R3 DECREASE_DISTANCE → G2 减小 (-0.2)
    #   R键(cmd=4) = R4 INCREASE_DISTANCE → G2 增大 (+0.2)
    #
    # 测试参数：speed=10, dist=40 → gap=4.0s
    # G2=4.0 时 error=0, G2=3.8 时 error=+0.2, G2=4.2 时 error=-0.2
    Y0_G2 = 10.0
    SPD_G2 = 10.0
    DIST_G2 = 40.0

    # --- 4a: Baseline (G2 ≈ 4.0) ---
    # Phase 2-3 未发送 T/R 键，G2 应仍为 4.0
    val_base = measure_with_fresh_sppvt(client, Y0_G2, SPD_G2, DIST_G2, TGT_SPD)
    # error = 4.0 - 4.0 = 0.0, expected output ≈ Kp*0 + Y0 = Y0
    print(f"  Baseline (G2=4.0, error=0.0): output={val_base:.4f} (expected ~{Y0_G2})")

    # --- 4b: Press T key (G2 -= 0.2 → G2=3.8, 缩短时距) ---
    # T键(cmd=3) = R3 DECREASE_DISTANCE → G2_Manager: G2 -= 0.2
    client.call(make_input(3, SPD_G2, DIST_G2, TGT_SPD, 99.0))
    time.sleep(0.02)
    # 重新 cancel→activate→read, 隔离 SPPVT 动态
    val_after_t = measure_with_fresh_sppvt(client, Y0_G2, SPD_G2, DIST_G2, TGT_SPD)
    # error = 4.0 - 3.8 = +0.2, expected output > baseline
    print(f"  After T key (G2~3.8, error~+0.2): output={val_after_t:.4f}")

    if val_after_t > val_base + 0.01:
        print(f"  [PASS] T key: G2 decreased (缩短时距) -> error positive -> output increased")
    else:
        print(f"  [FAIL] T key: expected output > {val_base:.4f}, "
              f"got {val_after_t:.4f}")
        all_pass = False

    # --- 4c: Press R key twice (G2 = 3.8 + 0.2 + 0.2 = 4.2, 延长时距) ---
    # R键(cmd=4) = R4 INCREASE_DISTANCE → G2_Manager: G2 += 0.2
    client.call(make_input(4, SPD_G2, DIST_G2, TGT_SPD, 99.0))
    time.sleep(0.02)
    client.call(make_input(4, SPD_G2, DIST_G2, TGT_SPD, 99.0))
    time.sleep(0.02)
    val_after_r = measure_with_fresh_sppvt(client, Y0_G2, SPD_G2, DIST_G2, TGT_SPD)
    # error = 4.0 - 4.2 = -0.2, expected output < baseline
    print(f"  After 2x R key (G2~4.2, error~-0.2): output={val_after_r:.4f}")

    if val_after_r < val_base - 0.01:
        print(f"  [PASS] R key: G2 increased (延长时距) -> error negative -> output decreased")
    else:
        print(f"  [FAIL] R key: expected output < {val_base:.4f}, "
              f"got {val_after_r:.4f}")
        all_pass = False

    # --- 4d: Quantitative check ---
    # T key 后 error=+0.2, R key 后 error=-0.2, delta_error=0.4
    delta_output = val_after_t - val_after_r  # 正误差输出 - 负误差输出
    delta_error = 0.4  # (+0.2) - (-0.2)
    kp_from_g2 = delta_output / delta_error if delta_error > 0 else 0
    print(f"  Output shift: {delta_output:.4f} over error shift {delta_error}")
    print(f"  Kp estimated from G2 test: {kp_from_g2:.4f}")
    if kp is not None:
        print(f"  Kp from Phase 3a: {kp:.4f} "
              f"(match={abs(kp_from_g2 - kp) < 5.0})")

    # ================================================================
    # Phase 5: Stability Tests
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 5: Stability Tests")
    print("=" * 70)

    # --- 5a: control_enabled stability (constant input) ---
    print("\n  --- 5a: control_enabled stability (20 iterations) ---")
    # 恢复 G2 到 4.0: Phase 4 结束时 G2≈4.2, 按 T 一次(缩短时距) → 4.0
    client.call(make_input(3, 15.0, 60.0, TGT_SPD, 99.0))
    time.sleep(0.02)
    # Cancel → activate, 进入 S0
    client.call(make_input(7, 15.0, 60.0, TGT_SPD, 0.0))
    time.sleep(0.02)
    client.call(make_input(1, 15.0, 60.0, TGT_SPD, 10.0))
    time.sleep(0.02)

    stable_in = make_input(0, 15.0, 60.0, TGT_SPD, 10.0)
    stable_enabled = None
    toggles = 0
    for i in range(20):
        out = client.call(stable_in)
        cur_enabled = bool(int(round(out[0])))
        if stable_enabled is None:
            stable_enabled = cur_enabled
        elif cur_enabled != stable_enabled:
            toggles += 1
            if toggles <= 3:
                print(f"    mismatch at step {i+1}: "
                      f"enabled={cur_enabled} (expected {stable_enabled})")
        time.sleep(0.02)
    if toggles == 0:
        print(f"    [PASS] Stable: control_enabled={stable_enabled}")
    else:
        print(f"    [FAIL] Unstable: {toggles} toggles")
        all_pass = False

    # --- 5b: Output convergence at zero error ---
    print("\n  --- 5b: Output convergence at zero error (30 iterations) ---")
    print("    Conditions: speed=15, dist=60 → gap=4.0s, G2≈4.0 → error≈0")
    print("    At zero error, SPPVT should converge to ~Y0")
    # Cancel → activate to reset SPPVT
    client.call(make_input(7, 15.0, 60.0, TGT_SPD, 0.0))
    time.sleep(0.02)
    client.call(make_input(1, 15.0, 60.0, TGT_SPD, 10.0))
    time.sleep(0.02)

    zero_err_in = make_input(0, 15.0, 60.0, TGT_SPD, 10.0)
    outputs = []
    for i in range(30):
        out = client.call(zero_err_in)
        outputs.append(out[1])
        time.sleep(0.02)

    last10 = outputs[-10:]
    spread = max(last10) - min(last10)
    mean_last10 = sum(last10) / len(last10)
    print(f"    First 5:  {[round(v, 2) for v in outputs[:5]]}")
    print(f"    Last 5:   {[round(v, 2) for v in outputs[-5:]]}")
    print(f"    Last 10 spread: {spread:.4f}, mean: {mean_last10:.4f}")
    if spread < 5.0:
        print(f"    [PASS] Output converging (spread={spread:.4f})")
    else:
        print(f"    [FAIL] Output NOT converging at zero error: "
              f"spread={spread:.4f}")
        all_pass = False

    # --- 5c: Non-zero error divergence direction ---
    print("\n  --- 5c: Non-zero error direction check ---")
    # 正误差 (gap > G2): output 应该逐步增大（加速）
    client.call(make_input(7, 10.0, 50.0, TGT_SPD, 0.0))
    time.sleep(0.02)
    client.call(make_input(1, 10.0, 50.0, TGT_SPD, 10.0))
    time.sleep(0.02)
    pos_err_in = make_input(0, 10.0, 50.0, TGT_SPD, 10.0)  # gap=5.0, error=+1.0
    out_first = client.call(pos_err_in)
    time.sleep(0.02)
    for _ in range(9):
        client.call(pos_err_in)
        time.sleep(0.02)
    out_later = client.call(pos_err_in)
    time.sleep(0.02)
    print(f"    Positive error (gap=5.0 > G2=4.0): "
          f"first={out_first[1]:.2f}, after 10 steps={out_later[1]:.2f}")
    if out_later[1] > out_first[1]:
        print(f"    [PASS] Output increasing (correct: wants to speed up)")
    else:
        print(f"    [FAIL] Output should increase for positive error")
        all_pass = False

    # 负误差 (gap < G2): output 应该逐步减小（减速）
    client.call(make_input(7, 10.0, 30.0, TGT_SPD, 0.0))
    time.sleep(0.02)
    client.call(make_input(1, 10.0, 30.0, TGT_SPD, 10.0))
    time.sleep(0.02)
    neg_err_in = make_input(0, 10.0, 30.0, TGT_SPD, 10.0)  # gap=3.0, error=-1.0
    out_first = client.call(neg_err_in)
    time.sleep(0.02)
    for _ in range(9):
        client.call(neg_err_in)
        time.sleep(0.02)
    out_later = client.call(neg_err_in)
    time.sleep(0.02)
    print(f"    Negative error (gap=3.0 < G2=4.0): "
          f"first={out_first[1]:.2f}, after 10 steps={out_later[1]:.2f}")
    if out_later[1] < out_first[1]:
        print(f"    [PASS] Output decreasing (correct: wants to slow down)")
    else:
        print(f"    [FAIL] Output should decrease for negative error")
        all_pass = False

    client.cleanup()

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 70}")
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - check output above")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
