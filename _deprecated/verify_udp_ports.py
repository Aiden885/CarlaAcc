"""
验证UDP端口绑定和通信
"""
import socket
import struct

def test_port_binding():
    """测试Python是否成功绑定到9090/9091端口"""
    print("="*70)
    print("测试UDP端口绑定")
    print("="*70)

    # 测试9090端口
    print("\n测试端口9090（决策模型）:")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 9090))

        # 获取实际绑定的地址
        actual_addr = sock.getsockname()
        print(f"  ✅ 成功绑定: {actual_addr[0]}:{actual_addr[1]}")

        # 发送测试数据到Simulink
        test_data = struct.pack('<4d', 2.0, 0.0, 0.0, 8.0)
        sock.sendto(test_data, ('127.0.0.1', 25000))
        print(f"  ✅ 成功发送测试数据到25000")

        sock.close()
    except Exception as e:
        print(f"  ❌ 失败: {e}")

    # 测试9091端口
    print("\n测试端口9091（SPPVT模型）:")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 9091))

        actual_addr = sock.getsockname()
        print(f"  ✅ 成功绑定: {actual_addr[0]}:{actual_addr[1]}")

        # 发送测试数据到Simulink
        test_data = struct.pack('<5d', 0.0, 0.0, 0.0, 0.0, 0.0)
        sock.sendto(test_data, ('127.0.0.1', 26000))
        print(f"  ✅ 成功发送测试数据到26000")

        sock.close()
    except Exception as e:
        print(f"  ❌ 失败: {e}")

def test_udp_receive():
    """测试是否能从Simulink接收数据"""
    print("\n" + "="*70)
    print("测试接收Simulink返回的数据")
    print("="*70)

    print("\n⚠️  确保Simulink模型正在运行！")

    # 测试决策模型
    print("\n测试决策模型（25000 → 25001）:")
    try:
        # 发送socket
        sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock_send.bind(('127.0.0.1', 9090))

        # 接收socket
        sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock_recv.bind(('127.0.0.1', 25001))
        sock_recv.settimeout(2.0)

        # 发送数据
        test_data = struct.pack('<4d', 2.0, 1.0, 0.0, 8.0)  # S2 + E键
        sock_send.sendto(test_data, ('127.0.0.1', 25000))
        print(f"  ✅ 已发送: state=2, cmd=1 (E键)")

        # 尝试接收
        try:
            recv_data, addr = sock_recv.recvfrom(1024)
            outputs = struct.unpack('<5d', recv_data)
            print(f"  ✅ 收到响应: state={outputs[0]}, decision={outputs[1]}, enabled={outputs[2]}")

            if outputs[0] == 2.0 and outputs[1] == 8.0:
                print(f"  ⚠️  警告：输出是默认值！Simulink可能没收到数据")
            elif outputs[0] == 0.0 and outputs[1] == 5.0 and outputs[2] == 1.0:
                print(f"  ✅ 正确！Simulink正确处理了E键")
            else:
                print(f"  ⚠️  输出异常")
        except socket.timeout:
            print(f"  ❌ 超时：没有收到Simulink的响应")

        sock_send.close()
        sock_recv.close()

    except Exception as e:
        print(f"  ❌ 失败: {e}")

if __name__ == '__main__':
    test_port_binding()
    test_udp_receive()

    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("如果看到'输出是默认值'的警告，说明：")
    print("  1. Simulink收不到Python发送的数据")
    print("  2. 可能是Remote port配置问题")
    print("  3. 或者Output latest data在非阻塞模式下一直输出初始值")
    print("\n建议：")
    print("  1. 在Simulink中取消Remote port验证")
    print("  2. 或者改回阻塞模式，但确保持续发送数据")
