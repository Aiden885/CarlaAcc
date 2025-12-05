"""
测试UDP端口是否可用
检查26000和26001端口的状态
"""
import socket
import struct
import time

def test_port_binding(port, description):
    """测试是否能绑定到指定端口"""
    print(f"\n测试端口 {port} ({description}):")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))
        print(f"  ✅ 端口{port}可用，可以绑定")
        sock.close()
        return True
    except OSError as e:
        print(f"  ❌ 端口{port}不可用: {e}")
        print(f"     可能已被其他程序占用")
        return False

def test_send_to_simulink():
    """尝试发送数据到Simulink"""
    print(f"\n测试发送数据到SPPVT模型 (端口26000):")

    try:
        # 创建发送socket
        sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 创建接收socket
        sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 尝试绑定接收端口
        try:
            sock_recv.bind(('127.0.0.1', 26001))
            print(f"  ✅ 成功绑定接收端口26001")
        except OSError as e:
            print(f"  ❌ 无法绑定端口26001: {e}")
            print(f"     如果Simulink正在运行，这是正常的（Simulink占用了26001）")
            print(f"     但这意味着我们无法接收Simulink的响应")
            sock_send.close()
            sock_recv.close()
            return False

        sock_recv.settimeout(2.0)

        # 发送测试数据（5个零）
        test_data = struct.pack('<5d', 0.0, 0.0, 0.0, 0.0, 0.0)
        print(f"  📤 发送测试数据到 127.0.0.1:26000...")
        sock_send.sendto(test_data, ('127.0.0.1', 26000))

        # 尝试接收响应
        print(f"  ⏳ 等待Simulink响应（2秒超时）...")
        try:
            resp, addr = sock_recv.recvfrom(1024)
            print(f"  ✅ 收到响应！{len(resp)}字节，来自{addr}")

            # 解析响应
            if len(resp) == 40:  # 5个double
                outputs = struct.unpack('<5d', resp)
                print(f"     输出: {outputs}")
                print(f"  ✅ SPPVT模型UDP通信正常！")
                success = True
            else:
                print(f"  ⚠️  响应大小不对，期望40字节，实际{len(resp)}字节")
                success = False

        except socket.timeout:
            print(f"  ❌ 超时：没有收到Simulink的响应")
            print(f"     可能原因:")
            print(f"     1. Simulink模型没有运行")
            print(f"     2. UDP Send端口配置错误（应该是26001）")
            print(f"     3. 模型内部出错，停止发送数据")
            success = False

        sock_send.close()
        sock_recv.close()
        return success

    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False

def check_simulink_status():
    """检查Simulink是否在监听26000端口"""
    print(f"\n检查Simulink是否在监听端口:")

    # 尝试连接到26000（虽然UDP不需要连接，但可以检测端口）
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.1)

    # 发送探测包
    try:
        sock.sendto(b'ping', ('127.0.0.1', 26000))
        print(f"  ✅ 可以发送到端口26000（Simulink可能在监听）")
    except Exception as e:
        print(f"  ⚠️  发送到26000失败: {e}")

    sock.close()

def main():
    print("="*70)
    print("UDP端口测试工具")
    print("="*70)

    print("\n⚠️  注意: 运行此测试前，请确保:")
    print("  1. Simulink模型已经启动（如果要测试通信）")
    print("  2. 没有其他程序占用26000/26001端口")

    # 测试端口可用性
    print("\n" + "="*70)
    print("第1步: 测试端口可用性")
    print("="*70)

    # 注意：如果Simulink正在运行，26001会被占用，这是正常的
    port_26000_ok = test_port_binding(26000, "Python→Simulink")
    port_26001_ok = test_port_binding(26001, "Simulink→Python")

    if not port_26000_ok:
        print(f"\n❌ 端口26000被占用，无法继续测试")
        print(f"   请检查是否有其他程序使用此端口:")
        print(f"   Linux: lsof -i :26000")
        print(f"   Windows: netstat -ano | findstr 26000")
        return

    if not port_26001_ok:
        print(f"\n⚠️  端口26001被占用")
        print(f"   如果Simulink正在运行，这是正常的")
        print(f"   否则请检查是否有其他程序占用此端口")

    # 检查Simulink状态
    print("\n" + "="*70)
    print("第2步: 检查Simulink监听状态")
    print("="*70)
    check_simulink_status()

    # 测试实际通信
    print("\n" + "="*70)
    print("第3步: 测试实际UDP通信")
    print("="*70)
    print("⚠️  此步骤需要Simulink模型正在运行")

    input("\n按回车键继续测试（确保Simulink已启动）...")

    if test_send_to_simulink():
        print("\n" + "="*70)
        print("✅ 所有测试通过！UDP通信正常")
        print("="*70)
        print("\n可以运行主程序:")
        print("  python test_udp_stress.py")
        print("  python acc_updated.py")
    else:
        print("\n" + "="*70)
        print("❌ UDP通信测试失败")
        print("="*70)
        print("\n请在MATLAB中运行诊断脚本:")
        print("  run diagnose_sppvt_udp.m")
        print("\n检查:")
        print("  1. Simulink模型是否正在运行")
        print("  2. UDP Receive端口是否配置为26000")
        print("  3. UDP Send端口是否配置为26001")

if __name__ == '__main__':
    main()
