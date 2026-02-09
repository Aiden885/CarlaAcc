#!/usr/bin/env python3
"""
End-to-end Decision+SPPVT UDP model test (acc_integrated_model).

This script sends raw 7-element input vectors directly to the Simulink
integrated model and validates Decision outputs against state_transitions.json.

Prerequisites:
  1) In MATLAB: run start_integrated_simulink_server.m
  2) Ensure decision_lookup_data.mat is loaded into the model workspace
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

    # decision = -1 means "use last_active_decision"
    if decision == -1:
        decision = int(last_decision)

    # has_history side-effect
    next_has_history = bool(has_history)
    if rule.get("side_effect") == "set_has_history_true":
        next_has_history = True

    # next_last_decision update: only update when decision in [1..6]
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


def run_cases():
    transitions = load_transitions()

    client = SimulinkUDPClient(
        send_port=27000,
        recv_port=27001,
        num_inputs=7,
        num_outputs=6,
        timeout=1.0,
        debug=False,
        local_send_port=9090,
        send_initial_packet=False,
    )

    # Test sequence: (state, cmd, has_history, last_decision, control_error, y0, reset_flag)
    cases = [
        ("S2 idle",                2, 0, 0, 8, 0.1, 10.0, 0),
        ("S2 + E (activate)",      2, 1, 0, 8, 0.1, 10.0, 0),
        ("S0 keep (cmd0)",         0, 0, 1, 5, 0.2, 10.0, 0),
        ("S0 cancel (cmd7)",       0, 7, 1, 5, 0.2, 10.0, 0),
        ("S1 idle",                1, 0, 1, 5, 0.2, 10.0, 0),
        ("S1 + Q (inherit)",       1, 2, 1, 5, 0.2, 10.0, 0),
        ("S0 + W (torque arb)",    0, 5, 1, 6, 0.2, 10.0, 0),
        ("S0 + S (exit)",          0, 6, 1, 6, 0.2, 10.0, 0),
        ("S1 + E (activate)",      1, 1, 1, 6, 0.2, 10.0, 0),
        ("S1 + E w/ reset_flag",   1, 1, 1, 6, 0.2, 10.0, 1),
    ]

    print("=== Integrated Model Decision Test ===")
    print("Inputs: [current_state, command_type, has_history, last_active_decision, control_error, Y0, reset_flag]")
    print()

    passed = 0
    failed = 0

    for name, st, cmd, hist, last_dec, err, y0, reset_flag in cases:
        inputs = [float(st), float(cmd), float(hist), float(last_dec), float(err), float(y0), float(reset_flag)]

        outputs = client.call(inputs)
        # outputs = [next_state, decision, control_enabled, next_has_history, next_last_decision, control_output]
        actual = {
            "next_state": int(round(outputs[0])),
            "decision": int(round(outputs[1])),
            "control_enabled": bool(int(round(outputs[2]))),
            "next_has_history": bool(int(round(outputs[3]))),
            "next_last_decision": int(round(outputs[4])),
        }

        expected = expected_transition(st, cmd, hist, last_dec, transitions)

        ok = actual == expected
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        print(f"  input:  state={st} cmd={cmd} hist={hist} last={last_dec} reset={reset_flag}")
        print(f"  expect: {expected}")
        print(f"  actual: {actual}")
        print()

        if ok:
            passed += 1
        else:
            failed += 1

        time.sleep(0.02)

    print(f"Summary: {passed} passed, {failed} failed.")

    # ------------------------
    # Stability test: constant input
    # ------------------------
    print("\n=== Stability Test: constant input (S2 cmd0) ===")
    stable_in = [2.0, 0.0, 0.0, 8.0, 0.1, 10.0, 0.0]
    stable_out = None
    toggles = 0
    for i in range(20):
        out = client.call(stable_in)
        cur = (int(round(out[0])), int(round(out[1])), int(round(out[2])))
        if stable_out is None:
            stable_out = cur
        elif cur != stable_out:
            toggles += 1
            print(f"  ⚠️ mismatch at step {i+1}: {cur} (expected {stable_out})")
        time.sleep(0.02)
    if toggles == 0:
        print("  ✅ stable output for constant input")
    else:
        print(f"  ❌ unstable output: {toggles} mismatches")

    # ------------------------
    # Closed-loop test: activate once, then hold cmd0
    # ------------------------
    print("\n=== Closed-loop Test: activate then hold ===")
    state = 2
    hist = 0
    last_dec = 8
    cmd_seq = [1] + [0] * 10
    for i, cmd in enumerate(cmd_seq):
        inputs = [float(state), float(cmd), float(hist), float(last_dec), 0.2, 10.0, 0.0]
        out = client.call(inputs)
        state = int(round(out[0]))
        decision = int(round(out[1]))
        enabled = bool(int(round(out[2])))
        hist = int(round(out[3]))
        last_dec = int(round(out[4]))
        print(f"  step {i+1:02d}: cmd={cmd} -> state={state} decision={decision} enabled={enabled}")

    # ------------------------
    # Extended closed-loop test: 100 iterations with realistic values
    # ------------------------
    print("\n=== Extended Closed-loop Test (100 iterations) ===")
    state = 2
    hist = 0
    last_dec = 8
    oscillation_count = 0
    prev_state = state

    # Simulate: E key to activate, then hold with realistic error/Y0 values
    for i in range(100):
        cmd = 1 if i == 0 else 0  # E key only on first iteration
        error = 0.5 + 0.1 * (i % 10)  # Varying error like in actual runtime
        y0 = 15.0 + 0.5 * (i % 5)     # Varying Y0

        inputs = [float(state), float(cmd), float(hist), float(last_dec), error, y0, 0.0]
        out = client.call(inputs)
        state = int(round(out[0]))
        decision = int(round(out[1]))
        enabled = bool(int(round(out[2])))
        hist = int(round(out[3]))
        last_dec = int(round(out[4]))

        # Detect state oscillation
        if state != prev_state:
            oscillation_count += 1
            if oscillation_count <= 5:  # Only print first 5
                print(f"  step {i+1:03d}: state changed {prev_state} -> {state} (decision={decision})")
        prev_state = state

        time.sleep(0.02)

    if oscillation_count <= 1:  # Allow one transition (S2 -> S0 on activation)
        print(f"  ✅ Stable: only {oscillation_count} state change(s)")
    else:
        print(f"  ❌ Oscillation detected: {oscillation_count} state changes")

    client.cleanup()

    # ------------------------
    # Test with send_initial_packet=True (like actual runtime)
    # ------------------------
    print("\n=== Test with send_initial_packet=True ===")
    initial_values = [2.0, 0.0, 0.0, 8.0, 0.0, 10.0, 0.0]  # Initial S2 state
    client2 = SimulinkUDPClient(
        send_port=27000,
        recv_port=27001,
        num_inputs=7,
        num_outputs=6,
        timeout=1.0,
        debug=False,
        local_send_port=9091,  # Different port to avoid conflict
        send_initial_packet=True,
        initial_values=initial_values,
    )

    state = 2
    hist = 0
    last_dec = 8
    oscillation_count = 0
    prev_state = state

    for i in range(50):
        cmd = 1 if i == 0 else 0
        error = 0.5 + 0.1 * (i % 10)
        y0 = 15.0 + 0.5 * (i % 5)

        inputs = [float(state), float(cmd), float(hist), float(last_dec), error, y0, 0.0]
        out = client2.call(inputs)
        state = int(round(out[0]))
        decision = int(round(out[1]))
        enabled = bool(int(round(out[2])))
        hist = int(round(out[3]))
        last_dec = int(round(out[4]))

        if state != prev_state:
            oscillation_count += 1
            if oscillation_count <= 5:
                print(f"  step {i+1:03d}: state changed {prev_state} -> {state} (decision={decision})")
        prev_state = state

        time.sleep(0.02)

    if oscillation_count <= 1:
        print(f"  ✅ Stable: only {oscillation_count} state change(s)")
    else:
        print(f"  ❌ Oscillation detected: {oscillation_count} state changes")

    client2.cleanup()


if __name__ == "__main__":
    run_cases()
