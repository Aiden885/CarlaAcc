#!/usr/bin/env python3
"""
End-to-end test for acc_integrated_model with Error_Calculation + Y0_Latch subsystems.

Tests:
  1. Basic connectivity (10 inputs -> 6 outputs)
  2. Decision lookup table (state transitions unchanged)
  3. Error_Calculation correctness (time-gap mode)
  4. Y0_Latch behaviour, stability

Input vector (10 elements):
  [current_state, command_type, has_history, last_active_decision,
   ego_speed_ms, vehicle_distance, G2_s, target_speed_ms,
   current_engine_torque, reset_flag]

  Note: target_speed_ms is lead vehicle speed, for display only.
  Note: current_engine_torque is latched by Y0_Latch at ACC activation
        (control_enabled rising edge) and held as Y0 until next activation.

Output vector (6 elements):
  [next_state, decision, control_enabled, next_has_history,
   next_last_decision, control_output]

Prerequisites:
  1) In MATLAB: run start_integrated_simulink_server.m
  2) Ensure decision_lookup_data.mat is loaded into the model workspace
  3) Ensure Error_Calculation and Y0_Latch subsystems are wired in the model
"""
import json
import time
from pathlib import Path

from simulink_udp_interface import SimulinkUDPClient


ROOT = Path(__file__).resolve().parent
TRANSITIONS_PATH = ROOT / "state_transitions.json"


def load_transitions():
    with open(TRANSITIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["transitions"]


def expected_transition(state, cmd, has_history, last_decision, transitions):
    st = transitions.get(str(state), {})
    rule = st.get(str(cmd), st.get("default", {"next_state": 2, "decision": 8, "control_enabled": False}))

    next_state = int(rule.get("next_state", 2))
    decision = int(rule.get("decision", 8))
    control_enabled = bool(rule.get("control_enabled", False))

    if decision == -1:
        decision = int(last_decision)

    next_has_history = bool(has_history)
    if rule.get("side_effect") == "set_has_history_true":
        next_has_history = True

    if 1 <= decision <= 6:
        next_last_decision = int(decision)
    else:
        next_last_decision = int(last_decision)

    return {
        "next_state": next_state,
        "decision": decision,
        "control_enabled": control_enabled,
        "next_has_history": next_has_history,
        "next_last_decision": next_last_decision,
    }


def make_input(state, cmd, hist, last_dec, speed, dist, g2, tgt_spd, engine_torque, reset):
    """构造10元素输入向量"""
    return [float(x) for x in [state, cmd, hist, last_dec, speed, dist, g2, tgt_spd, engine_torque, reset]]


def compute_expected_error(speed, dist, g2):
    """
    计算 Error_Calculation 子系统应输出的误差值（固定时距模式）

    error = dist / max(speed, 0.1) - g2
    """
    safe_speed = max(speed, 0.1)
    actual_gap = dist / safe_speed
    return actual_gap - g2


def latch_y0(client, desired_y0, tgt_spd=12.0):
    """
    辅助函数：强制 Y0_Latch 锁存指定值。

    1. 先取消控制 (S0 + cmd=7) → control_enabled 变 False
    2. 再激活控制 (S1 + cmd=1 + engine_torque=desired_y0) → 上升沿锁存

    返回激活帧的输出。
    """
    # Step 1: 取消控制 → S1, control_enabled=False
    inp_cancel = make_input(0, 7, 1, 5, 15.0, 30.0, 2.0, tgt_spd, desired_y0, 0)
    client.call(inp_cancel)
    time.sleep(0.02)

    # Step 2: 激活控制 → S0, control_enabled=True (上升沿 → 锁存 engine_torque)
    inp_activate = make_input(1, 1, 1, 5, 15.0, 30.0, 2.0, tgt_spd, desired_y0, 0)
    out = client.call(inp_activate)
    time.sleep(0.02)

    return out


def run_all_tests():
    transitions = load_transitions()

    client = SimulinkUDPClient(
        send_port=27000,
        recv_port=27001,
        num_inputs=10,
        num_outputs=6,
        timeout=2.0,
        debug=False,
        local_send_port=9090,
        send_initial_packet=False,
    )

    all_pass = True

    # target_speed_ms: 前车速度，不参与计算，仅显示用，测试中给个固定值
    TGT_SPD = 12.0

    # ================================================================
    # Phase 1: Basic Connectivity
    # ================================================================
    print("=" * 70)
    print("Phase 1: Basic Connectivity (10 inputs -> 6 outputs)")
    print("=" * 70)

    test_in = make_input(2, 0, 0, 8, 15.0, 30.0, 2.0, TGT_SPD, 10.0, 0)
    try:
        out = client.call(test_in)
        if len(out) == 6 and all(abs(v) < 1e6 for v in out):
            print(f"  [PASS] Got 6 outputs: {[round(v, 4) for v in out]}")
        else:
            print(f"  [FAIL] Unexpected output: {out}")
            all_pass = False
    except Exception as e:
        print(f"  [FAIL] Connection error: {e}")
        client.cleanup()
        return

    time.sleep(0.02)

    # ================================================================
    # Phase 2: Decision State Transitions
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 2: Decision State Transitions (lookup table verification)")
    print("=" * 70)

    SPD, DIST, G2 = 15.0, 30.0, 2.0

    cases = [
        ("S2 idle",                2, 0, 0, 8),
        ("S2 + E (activate)",      2, 1, 0, 8),
        ("S0 keep (cmd0)",         0, 0, 1, 5),
        ("S0 cancel (cmd7)",       0, 7, 1, 5),
        ("S1 idle",                1, 0, 1, 5),
        ("S1 + Q (inherit)",       1, 2, 1, 5),
        ("S0 + W (torque arb)",    0, 5, 1, 6),
        ("S0 + S (exit)",          0, 6, 1, 6),
        ("S1 + E (activate)",      1, 1, 1, 6),
        ("S1 + E w/ reset_flag",   1, 1, 1, 6),
    ]

    passed = 0
    failed = 0

    for i, (name, st, cmd, hist, last_dec) in enumerate(cases):
        reset = 1.0 if "reset_flag" in name else 0.0
        inputs = make_input(st, cmd, hist, last_dec, SPD, DIST, G2, TGT_SPD, 10.0, reset)

        out = client.call(inputs)
        actual = {
            "next_state": int(round(out[0])),
            "decision": int(round(out[1])),
            "control_enabled": bool(int(round(out[2]))),
            "next_has_history": bool(int(round(out[3]))),
            "next_last_decision": int(round(out[4])),
        }

        expected = expected_transition(st, cmd, hist, last_dec, transitions)
        ok = actual == expected
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            print(f"    expect: {expected}")
            print(f"    actual: {actual}")
            all_pass = False
            failed += 1
        else:
            passed += 1
        time.sleep(0.02)

    print(f"  Summary: {passed} passed, {failed} failed")

    # ================================================================
    # Phase 3: Error Calculation + Y0_Latch Verification
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 3: Error Calculation + Y0_Latch Verification")
    print("=" * 70)

    # All error tests: S0 (control active), cmd=0, reset_flag=1
    # With reset_flag=1, SPPVT stage_offset=0, so Y = Kp * error + Y0_latched

    ST, CMD, HIST, LAST_DEC = 0, 0, 1, 5

    # --- 3a: Extract Kp ---
    print("\n  --- 3a: Extract Kp (varying distance) ---")

    y0_test = 10.0
    speed_test = 10.0
    g2_test = 2.0

    # 先锁存 Y0 = y0_test
    latch_y0(client, y0_test, TGT_SPD)
    print(f"    Y0_Latch: latched engine_torque = {y0_test}")

    # Case 1: dist=20 -> gap=2.0, error=0.0
    # Case 2: dist=30 -> gap=3.0, error=1.0
    inp1 = make_input(ST, CMD, HIST, LAST_DEC, speed_test, 20.0, g2_test, TGT_SPD, 99.0, 1)
    inp2 = make_input(ST, CMD, HIST, LAST_DEC, speed_test, 30.0, g2_test, TGT_SPD, 99.0, 1)
    # 注意：engine_torque 传 99.0 是故意的，验证已锁存的 Y0 不受后续输入影响

    out1 = client.call(inp1)
    time.sleep(0.02)
    out2 = client.call(inp2)
    time.sleep(0.02)

    y1 = out1[5]
    y2 = out2[5]
    e1 = compute_expected_error(speed_test, 20.0, g2_test)  # 0.0
    e2 = compute_expected_error(speed_test, 30.0, g2_test)  # 1.0

    print(f"    Case 1: dist=20, expected_error={e1:.2f}, Y={y1:.4f}")
    print(f"    Case 2: dist=30, expected_error={e2:.2f}, Y={y2:.4f}")

    kp = None
    if abs(e2 - e1) < 1e-9:
        print("    [FAIL] Cannot extract Kp: errors are identical")
        all_pass = False
    else:
        kp = (y2 - y1) / (e2 - e1)
        y0_actual = y1 - kp * e1
        print(f"    Extracted: Kp={kp:.4f}, Y0_from_output={y0_actual:.4f} (latched Y0={y0_test})")

        if abs(y0_actual - y0_test) < 0.5:
            print(f"    [PASS] Y0 match (diff={abs(y0_actual - y0_test):.4f})")
        else:
            print(f"    [WARN] Y0 mismatch: expected ~{y0_test}, got {y0_actual:.4f}")

    # --- 3b: Linearity (varying distance) ---
    print("\n  --- 3b: Linearity (varying distance) ---")
    if kp is not None:
        test_dists = [10.0, 15.0, 25.0, 40.0, 50.0]
        verify_passed = 0
        for dist in test_dists:
            inp = make_input(ST, CMD, HIST, LAST_DEC, speed_test, dist, g2_test, TGT_SPD, 99.0, 1)
            out = client.call(inp)
            time.sleep(0.02)
            y_actual = out[5]
            e_expected = compute_expected_error(speed_test, dist, g2_test)
            y_expected = kp * e_expected + y0_test
            diff = abs(y_actual - y_expected)
            ok = diff < 0.5
            status = "PASS" if ok else "FAIL"
            print(f"    dist={dist:5.1f}: error={e_expected:6.2f}, Y_expect={y_expected:8.4f}, Y_actual={y_actual:8.4f}, diff={diff:.4f} [{status}]")
            if ok:
                verify_passed += 1
            else:
                all_pass = False
        print(f"    Linearity: {verify_passed}/{len(test_dists)} passed")

    # --- 3c: Varying speed (same distance) ---
    print("\n  --- 3c: Varying speed (dist=30, G2=2.0) ---")
    if kp is not None:
        verify_passed = 0
        for spd in [5.0, 8.0, 12.0, 15.0, 20.0]:
            inp = make_input(ST, CMD, HIST, LAST_DEC, spd, 30.0, g2_test, TGT_SPD, 99.0, 1)
            out = client.call(inp)
            time.sleep(0.02)
            y_actual = out[5]
            e_expected = compute_expected_error(spd, 30.0, g2_test)
            y_expected = kp * e_expected + y0_test
            diff = abs(y_actual - y_expected)
            ok = diff < 0.5
            status = "PASS" if ok else "FAIL"
            print(f"    speed={spd:5.1f}: gap={30.0/max(spd,0.1):.2f}s, error={e_expected:6.2f}, Y_expect={y_expected:8.4f}, Y_actual={y_actual:8.4f} [{status}]")
            if ok:
                verify_passed += 1
            else:
                all_pass = False
        print(f"    Speed variation: {verify_passed}/5 passed")

    # --- 3d: target_speed_ms does NOT affect error ---
    print("\n  --- 3d: target_speed_ms does NOT affect control output ---")
    inp_tgt0  = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 30.0, 2.0, 0.0,  99.0, 1)
    inp_tgt20 = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 30.0, 2.0, 20.0, 99.0, 1)
    out_tgt0 = client.call(inp_tgt0)
    time.sleep(0.02)
    out_tgt20 = client.call(inp_tgt20)
    time.sleep(0.02)
    diff_tgt = abs(out_tgt0[5] - out_tgt20[5])
    if diff_tgt < 0.01:
        print(f"    [PASS] target_speed=0 vs 20: Y diff={diff_tgt:.6f} (no effect)")
    else:
        print(f"    [FAIL] target_speed affects output: diff={diff_tgt:.4f}")
        all_pass = False

    # --- 3e: Y0_Latch verification ---
    print("\n  --- 3e: Y0_Latch (latch captures engine_torque at activation) ---")

    # Latch with Y0 = 10
    latch_y0(client, 10.0, TGT_SPD)
    inp_ref = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 20.0, 2.0, TGT_SPD, 999.0, 1)
    out_y10 = client.call(inp_ref)
    time.sleep(0.02)
    print(f"    Latched Y0=10: output={out_y10[5]:.4f}")

    # Latch with Y0 = 20
    latch_y0(client, 20.0, TGT_SPD)
    out_y20 = client.call(inp_ref)
    time.sleep(0.02)
    print(f"    Latched Y0=20: output={out_y20[5]:.4f}")

    y_shift = out_y20[5] - out_y10[5]
    print(f"    Shift={y_shift:.4f} (expected ~10.0)")
    if abs(y_shift - 10.0) < 1.0:
        print(f"    [PASS] Y0_Latch shift correct")
    else:
        print(f"    [FAIL] Y0_Latch shift incorrect (expected 10.0, got {y_shift:.4f})")
        all_pass = False

    # --- 3e2: Verify latch HOLDS (changing input after latch has no effect) ---
    print("\n  --- 3e2: Y0_Latch holds (input change after latch has no effect) ---")
    latch_y0(client, 10.0, TGT_SPD)
    # 在控状态发送不同的 engine_torque，Y0 应保持 10.0 不变
    inp_hold1 = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 20.0, 2.0, TGT_SPD, 10.0, 1)
    inp_hold2 = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 20.0, 2.0, TGT_SPD, 50.0, 1)
    out_hold1 = client.call(inp_hold1)
    time.sleep(0.02)
    out_hold2 = client.call(inp_hold2)
    time.sleep(0.02)
    diff_hold = abs(out_hold1[5] - out_hold2[5])
    if diff_hold < 0.01:
        print(f"    [PASS] engine_torque 10 vs 50 while latched: Y diff={diff_hold:.6f} (latch holds)")
    else:
        print(f"    [FAIL] Y0_Latch did not hold: diff={diff_hold:.4f}")
        all_pass = False

    # --- 3f: Varying G2 ---
    print("\n  --- 3f: Varying G2 (dist=30, speed=10) ---")
    if kp is not None:
        # 重新锁存 Y0
        latch_y0(client, y0_test, TGT_SPD)
        verify_passed = 0
        for g2 in [1.0, 1.5, 2.5, 3.0]:
            inp = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 30.0, g2, TGT_SPD, 99.0, 1)
            out = client.call(inp)
            time.sleep(0.02)
            y_actual = out[5]
            e_expected = compute_expected_error(10.0, 30.0, g2)
            y_expected = kp * e_expected + y0_test
            diff = abs(y_actual - y_expected)
            ok = diff < 0.5
            status = "PASS" if ok else "FAIL"
            print(f"    G2={g2:.1f}: error={e_expected:6.2f}, Y_expect={y_expected:8.4f}, Y_actual={y_actual:8.4f} [{status}]")
            if ok:
                verify_passed += 1
            else:
                all_pass = False
        print(f"    G2 variation: {verify_passed}/4 passed")

    # ================================================================
    # Phase 4: Stability Tests
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 4: Stability Tests")
    print("=" * 70)

    # --- 4a: Constant input stability ---
    print("\n  --- 4a: Constant input (20 iterations) ---")
    stable_in = make_input(0, 0, 1, 5, 15.0, 30.0, 2.0, TGT_SPD, 10.0, 0)
    stable_out = None
    toggles = 0
    for i in range(20):
        out = client.call(stable_in)
        cur = (int(round(out[0])), int(round(out[1])), int(round(out[2])))
        if stable_out is None:
            stable_out = cur
        elif cur != stable_out:
            toggles += 1
            if toggles <= 3:
                print(f"    mismatch at step {i+1}: {cur} (expected {stable_out})")
        time.sleep(0.02)
    if toggles == 0:
        print(f"    [PASS] Stable (state={stable_out[0]}, decision={stable_out[1]}, enabled={stable_out[2]})")
    else:
        print(f"    [FAIL] Unstable: {toggles} mismatches")
        all_pass = False

    # --- 4b: Closed-loop: activate then hold ---
    print("\n  --- 4b: Closed-loop (activate then hold, 50 iterations) ---")
    state = 2
    hist = 0
    last_dec = 8
    osc = 0
    prev_state = state

    for i in range(50):
        cmd = 1 if i == 0 else 0
        spd = 15.0 + 0.5 * (i % 10)
        dist = 30.0 + 2.0 * (i % 10)
        et = 15.0  # fixed engine_torque (Y0 will be latched at activation)

        inp = make_input(state, cmd, hist, last_dec, spd, dist, 2.0, TGT_SPD, et, 0)
        out = client.call(inp)
        state = int(round(out[0]))
        decision = int(round(out[1]))
        enabled = bool(int(round(out[2])))
        hist = int(round(out[3]))
        last_dec = int(round(out[4]))

        if state != prev_state:
            osc += 1
            if osc <= 3:
                print(f"    step {i+1:03d}: state {prev_state} -> {state} (decision={decision})")
        prev_state = state
        time.sleep(0.02)

    if osc <= 1:
        print(f"    [PASS] Stable: {osc} state change(s)")
    else:
        print(f"    [FAIL] Oscillation: {osc} state changes")
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
