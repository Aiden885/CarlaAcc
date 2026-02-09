#!/usr/bin/env python3
"""
End-to-end test for acc_integrated_model with Error_Calculation subsystem.

Tests:
  1. Basic connectivity (11 inputs → 6 outputs)
  2. Decision lookup table (state transitions unchanged)
  3. Error_Calculation correctness (TIME/SPEED mode, no-target fallback)
  4. Mode switching verification

Input vector (11 elements):
  [current_state, command_type, has_history, last_active_decision,
   ego_speed_ms, vehicle_distance, G2_s, V_target_ms, control_mode_flag,
   Y0, reset_flag]

Output vector (6 elements):
  [next_state, decision, control_enabled, next_has_history,
   next_last_decision, control_output]

Prerequisites:
  1) In MATLAB: run start_integrated_simulink_server.m
  2) Ensure decision_lookup_data.mat is loaded into the model workspace
  3) Ensure Error_Calculation subsystem is wired in the model
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


def make_input(state, cmd, hist, last_dec, speed, dist, g2, vtgt, mode, y0, reset):
    """构造11元素输入向量"""
    return [float(x) for x in [state, cmd, hist, last_dec, speed, dist, g2, vtgt, mode, y0, reset]]


def compute_expected_error(speed, dist, g2, vtgt, mode):
    """
    计算 Error_Calculation 子系统应输出的误差值

    TIME mode (flag=1):
      有目标 (dist < 9995): error = dist/max(speed, 0.1) - g2
      无目标 (dist >= 9995): error = vtgt - speed
    SPEED mode (flag=2):
      error = vtgt - speed
    """
    if mode == 1:
        if dist >= 9995:
            return vtgt - speed
        else:
            safe_speed = max(speed, 0.1)
            actual_gap = dist / safe_speed
            return actual_gap - g2
    else:
        return vtgt - speed


def run_all_tests():
    transitions = load_transitions()

    client = SimulinkUDPClient(
        send_port=27000,
        recv_port=27001,
        num_inputs=11,
        num_outputs=6,
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
    print("Phase 1: Basic Connectivity (11 inputs -> 6 outputs)")
    print("=" * 70)

    test_in = make_input(2, 0, 0, 8, 15.0, 30.0, 2.0, 30.0, 1.0, 10.0, 0)
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

    SPD, DIST, G2, VTGT, MODE = 15.0, 30.0, 2.0, 30.0, 1.0

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
        inputs = make_input(st, cmd, hist, last_dec, SPD, DIST, G2, VTGT, MODE, 10.0, reset)

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
    # Phase 3: Error Calculation Verification
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 3: Error Calculation Verification")
    print("=" * 70)

    # All error tests use S0 (control active), cmd=0, reset_flag=1
    # With reset_flag=1, SPPVT stage_offset=0, so Y = Kp * error + Y0
    # We extract Kp from first pair, then verify all others.

    ST, CMD, HIST, LAST_DEC = 0, 0, 1, 5  # S0, keep, control active

    # --- 3a: Extract Kp using TIME mode ---
    print("\n  --- 3a: Extract Kp (TIME mode, varying distance) ---")

    y0_test = 10.0
    speed_test = 10.0
    g2_test = 2.0

    # Case 1: dist=20 → gap=2.0, error=0.0
    # Case 2: dist=30 → gap=3.0, error=1.0
    inp1 = make_input(ST, CMD, HIST, LAST_DEC, speed_test, 20.0, g2_test, 30.0, 1.0, y0_test, 1)
    inp2 = make_input(ST, CMD, HIST, LAST_DEC, speed_test, 30.0, g2_test, 30.0, 1.0, y0_test, 1)

    out1 = client.call(inp1)
    time.sleep(0.02)
    out2 = client.call(inp2)
    time.sleep(0.02)

    y1 = out1[5]  # control_output
    y2 = out2[5]
    e1 = compute_expected_error(speed_test, 20.0, g2_test, 30.0, 1.0)  # 0.0
    e2 = compute_expected_error(speed_test, 30.0, g2_test, 30.0, 1.0)  # 1.0

    print(f"    Case 1: dist=20, expected_error={e1:.2f}, Y={y1:.4f}")
    print(f"    Case 2: dist=30, expected_error={e2:.2f}, Y={y2:.4f}")

    if abs(e2 - e1) < 1e-9:
        print("    [FAIL] Cannot extract Kp: errors are identical")
        all_pass = False
        kp = None
    else:
        kp = (y2 - y1) / (e2 - e1)
        y0_actual = y1 - kp * e1
        print(f"    Extracted: Kp={kp:.4f}, Y0_from_output={y0_actual:.4f} (sent Y0={y0_test})")

        # Y0_from_output should be close to y0_test
        if abs(y0_actual - y0_test) < 0.5:
            print(f"    [PASS] Y0 match (diff={abs(y0_actual - y0_test):.4f})")
        else:
            print(f"    [WARN] Y0 mismatch: expected ~{y0_test}, got {y0_actual:.4f}")
            print(f"           (may indicate stage_offset is not exactly 0 after reset)")

    # --- 3b: Verify TIME mode with more distances ---
    print("\n  --- 3b: TIME mode linearity (varying distance) ---")
    if kp is not None:
        test_dists = [10.0, 15.0, 25.0, 40.0, 50.0]
        verify_passed = 0
        for dist in test_dists:
            inp = make_input(ST, CMD, HIST, LAST_DEC, speed_test, dist, g2_test, 30.0, 1.0, y0_test, 1)
            out = client.call(inp)
            time.sleep(0.02)
            y_actual = out[5]
            e_expected = compute_expected_error(speed_test, dist, g2_test, 30.0, 1.0)
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

    # --- 3c: SPEED mode verification ---
    # NOTE: SPPVT has output saturation at ~±800. With Kp=195, linear region
    # is roughly error in [-4.1, 4.1]. Use small speed errors to stay in range.
    print("\n  --- 3c: SPEED mode (small errors, within linear region) ---")
    if kp is not None:
        # speed=10, V_target close to 10 → small errors
        # Case 1: V_target=10 → error=0, Case 2: V_target=12 → error=2
        inp_s1 = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 30.0, 2.0, 10.0, 2.0, y0_test, 1)
        inp_s2 = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 30.0, 2.0, 12.0, 2.0, y0_test, 1)
        out_s1 = client.call(inp_s1)
        time.sleep(0.02)
        out_s2 = client.call(inp_s2)
        time.sleep(0.02)

        es1 = compute_expected_error(10.0, 30.0, 2.0, 10.0, 2.0)  # 0
        es2 = compute_expected_error(10.0, 30.0, 2.0, 12.0, 2.0)  # 2
        ys1 = out_s1[5]
        ys2 = out_s2[5]

        if abs(es2 - es1) < 1e-9:
            print("    [FAIL] Cannot verify: errors identical")
            all_pass = False
        else:
            kp_speed = (ys2 - ys1) / (es2 - es1)
            print(f"    V_target=10: error={es1:.1f}, Y={ys1:.4f}")
            print(f"    V_target=12: error={es2:.1f}, Y={ys2:.4f}")
            print(f"    Kp_speed={kp_speed:.4f} (Kp_time={kp:.4f})")

            # Kp should be the same for both modes (same SPPVT subsystem)
            if abs(kp_speed - kp) < 0.5:
                print(f"    [PASS] Kp consistent across modes")
            else:
                print(f"    [FAIL] Kp differs between modes (diff={abs(kp_speed - kp):.4f})")
                all_pass = False

            # Additional V_target values (keep error in [-4, 4])
            for vtgt in [7.0, 9.0, 11.0, 13.0]:
                inp = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 30.0, 2.0, vtgt, 2.0, y0_test, 1)
                out = client.call(inp)
                time.sleep(0.02)
                e_exp = compute_expected_error(10.0, 30.0, 2.0, vtgt, 2.0)
                y_exp = kp_speed * e_exp + y0_test
                y_act = out[5]
                diff = abs(y_act - y_exp)
                ok = diff < 0.5
                status = "PASS" if ok else "FAIL"
                print(f"    V_target={vtgt:5.1f}: error={e_exp:6.1f}, Y_expect={y_exp:8.4f}, Y_actual={y_act:8.4f} [{status}]")
                if not ok:
                    all_pass = False

    # --- 3d: No-target fallback ---
    # Use small error (V_target close to speed) to stay in linear region
    print("\n  --- 3d: No-target fallback (TIME mode, dist>=9995) ---")
    # TIME mode with dist=9999 should fall back to speed error = V_target - ego_speed
    # This should give the same result as SPEED mode with same speed/V_target
    inp_time_notarget = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 9999.0, 2.0, 12.0, 1.0, y0_test, 1)
    inp_speed_same    = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 9999.0, 2.0, 12.0, 2.0, y0_test, 1)

    out_tnt = client.call(inp_time_notarget)
    time.sleep(0.02)
    out_spd = client.call(inp_speed_same)
    time.sleep(0.02)

    y_time_notarget = out_tnt[5]
    y_speed = out_spd[5]

    e_time_notarget = compute_expected_error(10.0, 9999.0, 2.0, 12.0, 1.0)  # 12-10 = 2
    e_speed = compute_expected_error(10.0, 9999.0, 2.0, 12.0, 2.0)          # 12-10 = 2

    print(f"    TIME mode (no target): error={e_time_notarget:.1f}, Y={y_time_notarget:.4f}")
    print(f"    SPEED mode:            error={e_speed:.1f}, Y={y_speed:.4f}")
    diff_fallback = abs(y_time_notarget - y_speed)
    if diff_fallback < 0.01:
        print(f"    [PASS] No-target fallback matches SPEED mode (diff={diff_fallback:.6f})")
    else:
        print(f"    [FAIL] Mismatch: diff={diff_fallback:.4f}")
        all_pass = False

    # --- 3e: Mode switch produces different output ---
    print("\n  --- 3e: Mode switch (same inputs, different mode flag) ---")
    # speed=10, dist=30, G2=2.0, V_target=11
    # TIME: error = 30/10 - 2.0 = 1.0
    # SPEED: error = 11 - 10 = 1.0
    # Same error → same output (both in linear region)
    inp_same_err_time = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 30.0, 2.0, 11.0, 1.0, y0_test, 1)
    inp_same_err_spd  = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 30.0, 2.0, 11.0, 2.0, y0_test, 1)

    out_set = client.call(inp_same_err_time)
    time.sleep(0.02)
    out_ses = client.call(inp_same_err_spd)
    time.sleep(0.02)

    et_same = compute_expected_error(10.0, 30.0, 2.0, 11.0, 1.0)  # 1.0
    es_same = compute_expected_error(10.0, 30.0, 2.0, 11.0, 2.0)  # 1.0
    print(f"    Same-error test: TIME error={et_same:.1f}, SPEED error={es_same:.1f}")
    print(f"    TIME Y={out_set[5]:.4f}, SPEED Y={out_ses[5]:.4f}")
    diff_same = abs(out_set[5] - out_ses[5])
    if diff_same < 0.01:
        print(f"    [PASS] Same error → same output (diff={diff_same:.6f})")
    else:
        print(f"    [FAIL] Same error but different output (diff={diff_same:.4f})")
        all_pass = False

    # Different error: speed=10, dist=15, G2=2.0, V_target=12
    # TIME: error = 15/10 - 2.0 = -0.5
    # SPEED: error = 12 - 10 = 2.0
    inp_diff_time = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 15.0, 2.0, 12.0, 1.0, y0_test, 1)
    inp_diff_spd  = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 15.0, 2.0, 12.0, 2.0, y0_test, 1)

    out_dt = client.call(inp_diff_time)
    time.sleep(0.02)
    out_ds = client.call(inp_diff_spd)
    time.sleep(0.02)

    et_diff = compute_expected_error(10.0, 15.0, 2.0, 12.0, 1.0)  # -0.5
    es_diff = compute_expected_error(10.0, 15.0, 2.0, 12.0, 2.0)  # 2.0
    print(f"    Diff-error test: TIME error={et_diff:.1f}, SPEED error={es_diff:.1f}")
    print(f"    TIME Y={out_dt[5]:.4f}, SPEED Y={out_ds[5]:.4f}")
    diff_mode = abs(out_dt[5] - out_ds[5])
    if diff_mode > 0.1:
        print(f"    [PASS] Different errors → different outputs (diff={diff_mode:.4f})")
    else:
        print(f"    [FAIL] Should differ but nearly identical (diff={diff_mode:.6f})")
        all_pass = False

    # --- 3f: Y0 passthrough verification ---
    print("\n  --- 3f: Y0 passthrough (changing Y0 shifts output) ---")
    # Same error, different Y0 → output should shift by delta_Y0
    inp_y0_10 = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 20.0, 2.0, 30.0, 1.0, 10.0, 1)
    inp_y0_20 = make_input(ST, CMD, HIST, LAST_DEC, 10.0, 20.0, 2.0, 30.0, 1.0, 20.0, 1)

    out_y10 = client.call(inp_y0_10)
    time.sleep(0.02)
    out_y20 = client.call(inp_y0_20)
    time.sleep(0.02)

    y_out_10 = out_y10[5]
    y_out_20 = out_y20[5]
    y_shift = y_out_20 - y_out_10

    print(f"    Y0=10: output={y_out_10:.4f}")
    print(f"    Y0=20: output={y_out_20:.4f}")
    print(f"    Shift={y_shift:.4f} (expected ~10.0)")
    if abs(y_shift - 10.0) < 1.0:
        print(f"    [PASS] Y0 shift correct")
    else:
        print(f"    [FAIL] Y0 shift incorrect (expected 10.0, got {y_shift:.4f})")
        all_pass = False

    # ================================================================
    # Phase 4: Stability Test
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Phase 4: Stability Tests")
    print("=" * 70)

    # --- 4a: Constant input stability ---
    print("\n  --- 4a: Constant input (20 iterations) ---")
    stable_in = make_input(0, 0, 1, 5, 15.0, 30.0, 2.0, 30.0, 1.0, 10.0, 0)
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
        print(f"    [PASS] Stable output (state={stable_out[0]}, decision={stable_out[1]}, enabled={stable_out[2]})")
    else:
        print(f"    [FAIL] Unstable: {toggles} mismatches")
        all_pass = False

    # --- 4b: Closed-loop: activate then hold ---
    print("\n  --- 4b: Closed-loop (activate then hold cmd0, 50 iterations) ---")
    state = 2
    hist = 0
    last_dec = 8
    osc = 0
    prev_state = state

    for i in range(50):
        cmd = 1 if i == 0 else 0
        spd = 15.0 + 0.5 * (i % 10)
        dist = 30.0 + 2.0 * (i % 10)
        y0 = 15.0 + 0.5 * (i % 5)

        inp = make_input(state, cmd, hist, last_dec, spd, dist, 2.0, 30.0, 1.0, y0, 0)
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
