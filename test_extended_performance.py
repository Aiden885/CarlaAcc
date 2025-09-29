#!/usr/bin/env python3
"""
Extended performance test with more cycles to get accurate average timing
"""

import time
from acc_decision_sppvt_interface import ACCDecisionSPPVTInterface

def test_extended_performance():
    """Test multiple cycles to get accurate average timing"""
    print("=== Extended Performance Test (10 cycles) ===")

    # Create interface once
    print("Creating interface...")
    start_time = time.time()
    interface = ACCDecisionSPPVTInterface(debug=False, use_realtime_sppvt=True)  # 关闭debug减少日志
    init_time = time.time() - start_time
    print(f"Interface initialization: {init_time:.1f}s")

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

    print("\nRunning 10 cycles...")
    call_times = []
    sppvt_outputs = []

    # 预热调用
    print("Warmup call...")
    try:
        result = interface.process_decision_and_control(test_input)
        print(f"Warmup completed, SPPVT: {result.get('sppvt_control_output', 'N/A')}")
    except Exception as e:
        print(f"Warmup failed: {e}")
        return

    # 正式测试10个周期
    for i in range(10):
        print(f"Cycle {i+1}/10...", end=' ')

        # 稍微变化输入数据模拟真实场景
        test_input['control_error'] = 1.5 + i * 0.1
        test_input['ego_speed_ms'] = 13.89 + i * 0.5
        test_input['ego_speed_kmh'] = test_input['ego_speed_ms'] * 3.6
        test_input['timestamp'] = time.time()

        call_start = time.time()
        try:
            result = interface.process_decision_and_control(test_input)
            call_time = time.time() - call_start
            call_times.append(call_time)

            sppvt_output = result.get('sppvt_control_output', 0.0)
            sppvt_outputs.append(sppvt_output)

            print(f"{call_time*1000:.0f}ms, SPPVT: {sppvt_output:.3f}")

        except Exception as e:
            print(f"FAILED: {e}")
            call_times.append(None)
            sppvt_outputs.append(None)

    # 分析结果
    print(f"\n=== Performance Analysis ===")
    print(f"Initialization: {init_time:.1f}s")

    valid_times = [t for t in call_times if t is not None]
    valid_outputs = [o for o in sppvt_outputs if o is not None]

    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        min_time = min(valid_times)
        max_time = max(valid_times)

        print(f"\nTiming Statistics:")
        print(f"  Valid calls: {len(valid_times)}/10")
        print(f"  Average: {avg_time*1000:.0f}ms")
        print(f"  Min: {min_time*1000:.0f}ms")
        print(f"  Max: {max_time*1000:.0f}ms")
        print(f"  Range: {(max_time-min_time)*1000:.0f}ms")

        # 趋势分析
        if len(valid_times) >= 5:
            first_half = valid_times[:len(valid_times)//2]
            second_half = valid_times[len(valid_times)//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)

            print(f"\nTrend Analysis:")
            print(f"  First half avg: {first_avg*1000:.0f}ms")
            print(f"  Second half avg: {second_avg*1000:.0f}ms")
            if second_avg < first_avg:
                speedup = first_avg / second_avg
                print(f"  Performance improving: {speedup:.1f}x speedup")
            else:
                print(f"  Performance stable")

        # 目标评估
        target_time_ms = 500
        print(f"\nTarget Assessment:")
        print(f"  Target: <{target_time_ms}ms")
        print(f"  Current avg: {avg_time*1000:.0f}ms")
        if avg_time * 1000 <= target_time_ms:
            print(f"  ✅ TARGET ACHIEVED!")
        else:
            ratio = (avg_time * 1000) / target_time_ms
            print(f"  ❌ Need {ratio:.1f}x speedup to reach target")

    if valid_outputs:
        print(f"\nSPPVT Output Analysis:")
        print(f"  Outputs: {[f'{o:.3f}' for o in valid_outputs[:5]]}...")
        unique_outputs = len(set(f'{o:.3f}' for o in valid_outputs))
        print(f"  Unique values: {unique_outputs}/10")
        if unique_outputs > 1:
            print(f"  ✅ Output values changing (responsive)")
        else:
            print(f"  ⚠️ Output values constant")

    # Cleanup
    try:
        interface.cleanup()
    except:
        pass

if __name__ == "__main__":
    test_extended_performance()