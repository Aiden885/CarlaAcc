import argparse
import socket
import struct
import time
from typing import Tuple


def create_sockets(host: str, recv_port: int, timeout: float) -> Tuple[socket.socket, socket.socket]:
    """Create UDP send/recv sockets."""
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        recv_sock.bind((host, recv_port))
    except OSError as e:
        raise OSError(f"Bind failed on {host}:{recv_port} (maybe Simulink UDP Send is using it?): {e}") from e
    recv_sock.settimeout(timeout)
    return send_sock, recv_sock


def main():
    parser = argparse.ArgumentParser(description="UDP round-trip test for Simulink decision model.")
    parser.add_argument("--host", default="127.0.0.1", help="Simulink UDP host (default: 127.0.0.1)")
    parser.add_argument("--send-port", type=int, default=25000, help="Port Python sends to (Simulink UDP Receive local port)")
    parser.add_argument("--recv-port", type=int, default=25001, help="Port Python listens on (Simulink UDP Send remote port)")
    parser.add_argument("--count", type=int, default=5, help="Number of test frames")
    parser.add_argument("--period", type=float, default=0.05, help="Send period (s), match Simulink sample time")
    args = parser.parse_args()

    send_sock, recv_sock = create_sockets(args.host, args.recv_port, timeout=max(0.05, args.period * 2))

    fmt_out = "<4d"  # current_state, command_type, has_history, last_active_decision
    fmt_in = "<5d"   # next_state, decision, control_enabled, next_has_history, next_last_decision

    print(f"🌐 Testing decision UDP: send -> {args.host}:{args.send_port}, recv <- {args.host}:{args.recv_port}")
    print("   Payload out: current_state, command_type, has_history, last_active_decision (4 x float64)")
    print("   Payload in : next_state, decision, control_enabled, next_has_history, next_last_decision (5 x float64)")

    # Define test cases: (Description, Input, ExpectedOutput)
    # Input: (current_state, command_type, has_history, last_active_decision)
    # Expected: (next_state, decision, control_enabled) - use None if don't care
    test_cases = [
        {
            "desc": "S2(Standby) + Cmd1(E-Key) -> S0(Active), R5(CurrentSpeed)",
            "input": (2.0, 1.0, 0.0, 8.0),
            "expected": (0.0, 5.0, 1.0)
        },
        {
            "desc": "S0(Active) + Cmd0(None) -> S0(Active), R5(Hold/LastDec)",
            "input": (0.0, 0.0, 1.0, 5.0),
            "expected": (0.0, 5.0, 1.0) # Should use last_active_decision=5
        },
        {
            "desc": "S0(Active) + Cmd1(E-Key) -> S0(Active), R1(DecSpeed)",
            "input": (0.0, 1.0, 1.0, 5.0),
            "expected": (0.0, 1.0, 1.0)
        },
        {
            "desc": "S0(Active) + Cmd6(S-Key/Brake) -> S1(History), R8(Standby)",
            "input": (0.0, 6.0, 1.0, 5.0),
            "expected": (1.0, 8.0, 0.0)
        },
        {
            "desc": "S1(History) + Cmd2(Q-Key) -> S0(Active), R6(Inherit)",
            "input": (1.0, 2.0, 1.0, 8.0),
            "expected": (0.0, 6.0, 1.0)
        },
        {
            "desc": "S2(Standby) + Cmd0(None) -> S2(Standby), R8(Standby)",
            "input": (2.0, 0.0, 0.0, 8.0),
            "expected": (2.0, 8.0, 0.0)
        }
    ]

    passed_count = 0
    for i, test in enumerate(test_cases):
        payload = test["input"]
        data = struct.pack(fmt_out, *payload)
        send_sock.sendto(data, (args.host, args.send_port))

        try:
            resp, _ = recv_sock.recvfrom(1024)
            if len(resp) != struct.calcsize(fmt_in):
                print(f"[{i}] ⚠️ Unexpected packet size {len(resp)}, expected {struct.calcsize(fmt_in)}")
                continue
                
            next_state, decision, control_enabled, next_has_hist, next_last_dec = struct.unpack(fmt_in, resp)
            
            # Verify results
            exp_state, exp_dec, exp_enabled = test["expected"]
            
            fail_reasons = []
            if exp_state is not None and next_state != exp_state:
                fail_reasons.append(f"State: got {next_state} != exp {exp_state}")
            if exp_dec is not None and decision != exp_dec:
                fail_reasons.append(f"Decision: got {decision} != exp {exp_dec}")
            if exp_enabled is not None and int(control_enabled) != int(exp_enabled):
                fail_reasons.append(f"Enabled: got {int(control_enabled)} != exp {int(exp_enabled)}")
            
            status = "✅ PASS" if not fail_reasons else "❌ FAIL"
            print(f"\nTest {i+1}: {test['desc']}")
            print(f"   Input: S{payload[0]:.0f}, Cmd{payload[1]:.0f}, Hist{payload[2]:.0f}, Last{payload[3]:.0f}")
            print(f"   Output: S{next_state:.0f}, R{decision:.0f}, En{int(control_enabled)} | {status}")
            
            if fail_reasons:
                for reason in fail_reasons:
                    print(f"      -> {reason}")
            else:
                passed_count += 1

        except socket.timeout:
            print(f"\nTest {i+1}: {test['desc']}")
            print(f"   ❌ Timeout waiting for response")

        time.sleep(args.period)

    print(f"\nSummary: {passed_count}/{len(test_cases)} tests passed.")
    
    send_sock.close()
    recv_sock.close()
    print("✅ Decision UDP test completed.")


if __name__ == "__main__":
    main()
