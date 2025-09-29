#!/usr/bin/env python3
"""
Test multi-call performance to verify only first call is slow
"""

import time
from acc_decision_sppvt_interface import ACCDecisionSPPVTInterface

def test_multi_call_performance():
    """Test if subsequent calls are much faster"""
    print("=== Multi-call Performance Test ===")

    # Create interface once (this should be slow)
    print("Creating interface (expecting ~11 seconds)...")
    start_time = time.time()
    interface = ACCDecisionSPPVTInterface(debug=True, use_realtime_sppvt=True)
    init_time = time.time() - start_time
    print(f"Interface initialization: {init_time*1000:.1f}ms")

    # Prepare test input
    test_input = {
        'ego_speed_kmh': 50.0,
        'ego_speed_ms': 13.89,
        'command_type': 2,  # I1
        'command_active': True,
        'manual_throttle_active': False,
        'control_error': 1.5,
        'control_mode_flag': 2,  # speed mode
        'V_target_kmh': 50.0,
        'V_min_kmh': 30.0,
        'G2_s': 2.0,
        'timestamp': time.time(),
        'external_stage_offset': 0.0,
        'external_stage_manager_states': [1.0, -1.0, 2.0],
        'external_adapter_states': [1.0, 13.89, 0.2]
    }

    print("\nTesting multiple calls (expecting fast after first)...")

    call_times = []
    for i in range(5):
        print(f"\nCall {i+1}:")
        call_start = time.time()

        try:
            result = interface.process_decision_and_control(test_input)
            call_time = time.time() - call_start
            call_times.append(call_time)

            print(f"  Time: {call_time*1000:.1f}ms")
            print(f"  SPPVT: {result.get('sppvt_control_output', 'N/A')}")
            print(f"  State: S{result.get('current_state', 'N/A')}")
            print(f"  Decision: R{result.get('current_decision', 'N/A')}")

        except Exception as e:
            print(f"  FAILED: {e}")
            call_times.append(None)

    # Analyze results
    print(f"\n=== Performance Analysis ===")
    print(f"Initialization: {init_time*1000:.1f}ms")

    valid_times = [t for t in call_times if t is not None]
    if valid_times:
        print(f"Call times:")
        for i, t in enumerate(call_times):
            if t is not None:
                status = "✅ FAST" if t < 0.5 else "❌ SLOW"
                print(f"  Call {i+1}: {t*1000:.1f}ms {status}")
            else:
                print(f"  Call {i+1}: FAILED")

        if len(valid_times) > 1:
            first_call = valid_times[0]
            subsequent_avg = sum(valid_times[1:]) / len(valid_times[1:])
            print(f"\nFirst call: {first_call*1000:.1f}ms")
            print(f"Subsequent avg: {subsequent_avg*1000:.1f}ms")
            print(f"Speedup: {first_call/subsequent_avg:.1f}x")

            if subsequent_avg < 0.5:
                print("✅ SUCCESS: Subsequent calls are fast!")
                print("✅ The 11-second issue is only first-time initialization")
            else:
                print("❌ PROBLEM: Even subsequent calls are slow")
        else:
            print("❌ Not enough successful calls for comparison")

    # Cleanup
    try:
        interface.cleanup()
    except:
        pass

if __name__ == "__main__":
    test_multi_call_performance()