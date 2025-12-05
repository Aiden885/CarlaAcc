import argparse
import socket
import struct
import time
from typing import Tuple


def create_sockets(send_port: int, recv_port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> Tuple[socket.socket, socket.socket]:
    """Create a UDP sender and receiver socket bound to the given ports."""
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Allow quick rebinding if port was in TIME_WAIT
    recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        recv_sock.bind((host, recv_port))
    except OSError as e:
        raise OSError(f"Bind failed on {host}:{recv_port} (maybe Simulink or another test is already using it?): {e}") from e
    recv_sock.settimeout(timeout)
    return send_sock, recv_sock


def main():
    parser = argparse.ArgumentParser(description="Simple UDP round-trip test for SPPVT Simulink model.")
    parser.add_argument("--host", default="127.0.0.1", help="Simulink UDP host (default: 127.0.0.1)")
    parser.add_argument("--send-port", type=int, default=26000, help="Port Python sends to (Simulink UDP Receive local port)")
    parser.add_argument("--recv-port", type=int, default=26001, help="Port Python listens on (Simulink UDP Send remote port)")
    parser.add_argument("--count", type=int, default=5, help="Number of test frames to send")
    parser.add_argument("--period", type=float, default=0.05, help="Send period in seconds (match Simulink sample time)")
    args = parser.parse_args()

    send_sock, recv_sock = create_sockets(args.host, args.recv_port, timeout=max(0.05, args.period * 2))

    fmt_out = "<5d"  # error, stage_offset, prev_error, prev_vel, prev_accel
    fmt_in = "<5d"   # control_output, velocity_output, acceleration_output, jerk_output, should_upgrade

    print(f"🌐 Testing SPPVT UDP: send -> {args.host}:{args.send_port}, recv <- {args.host}:{args.recv_port}")
    print("   Payload format (out): error, stage_offset, prev_error, prev_vel, prev_accel (5 x float64)")
    print("   Payload format (in):  control, velocity, accel, jerk, should_upgrade (5 x float64)")

    for i in range(args.count):
        # Simple test pattern: vary error each frame, keep history terms zero
        error = 1.0 + 0.1 * i  # vary error to see response
        payload = (error, 0.0, 0.0, 0.0, 0.0)
        data = struct.pack(fmt_out, *payload)
        send_sock.sendto(data, (args.host, args.send_port))

        try:
            resp, _ = recv_sock.recvfrom(1024)
            if len(resp) != struct.calcsize(fmt_in):
                print(f"[{i}] ⚠️ Unexpected packet size {len(resp)}, expected {struct.calcsize(fmt_in)}")
                continue
            values = struct.unpack(fmt_in, resp)
            print(f"[{i}] sent error={error:.3f} -> control={values[0]:+.4f}, vel={values[1]:+.4f}, "
                  f"acc={values[2]:+.4f}, jerk={values[3]:+.4f}, upgrade={values[4]:.0f}")
        except socket.timeout:
            print(f"[{i}] ❌ Timeout waiting for response (is Simulink running with UDP Send to port {args.recv_port}?)")

        time.sleep(args.period)

    send_sock.close()
    recv_sock.close()
    print("✅ Test completed.")


if __name__ == "__main__":
    main()
