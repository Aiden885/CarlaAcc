"""
Simulink UDP通信接口
基于测试验证的UDP通信实现，替代MATLAB Engine
"""
import socket
import struct
import time
from typing import List, Tuple


class SimulinkUDPClient:
    """
    UDP客户端，与单个Simulink模型通信

    Args:
        send_port: 发送数据的目标端口（Simulink监听端口）
        recv_port: 接收数据的本地端口
        num_inputs: 输入参数数量
        num_outputs: 输出参数数量
        timeout: 接收超时时间（秒）
        debug: 是否打印调试信息
    """

    def __init__(self, send_port: int, recv_port: int,
                 num_inputs: int, num_outputs: int,
                 timeout: float = 0.1, debug: bool = False,
                 send_initial_packet: bool = True,
                 initial_values: list = None,
                 local_send_port: int = None):
        self.send_port = send_port
        self.recv_port = recv_port
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.timeout = timeout
        self.debug = debug
        self.host = '127.0.0.1'
        self.local_send_port = local_send_port

        # 创建UDP socket
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 允许发送socket快速重新绑定端口
        self.sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 允许快速重新绑定端口
        self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 绑定发送socket到固定源端口（如果指定）
        if local_send_port is not None:
            try:
                self.sock_send.bind((self.host, local_send_port))
                if self.debug:
                    print(f"✅ 发送socket绑定到本地端口: {local_send_port}")
            except OSError as e:
                raise OSError(
                    f"❌ 绑定发送端口{local_send_port}失败: {e}"
                ) from e

        # 绑定接收端口
        try:
            self.sock_recv.bind((self.host, recv_port))
        except OSError as e:
            raise OSError(
                f"❌ 绑定端口{recv_port}失败（可能Simulink未启动或端口被占用）: {e}"
            ) from e

        self.sock_recv.settimeout(timeout)

        # 性能统计
        self.call_count = 0
        self.total_time = 0.0

        # 数据包格式（little-endian double）
        self.fmt_send = f'<{num_inputs}d'
        self.fmt_recv = f'<{num_outputs}d'

        if self.debug:
            print(f"✅ UDP客户端初始化: 发送→{send_port}, 接收←{recv_port}")
            print(f"   输入: {num_inputs}个double, 输出: {num_outputs}个double")

        # 发送初始数据包，防止Simulink启动时UDP Receive超时
        if send_initial_packet:
            try:
                # 使用自定义初始值或全零
                if initial_values is not None:
                    if len(initial_values) != num_inputs:
                        raise ValueError(f"initial_values长度({len(initial_values)})与num_inputs({num_inputs})不匹配")
                    init_values = initial_values
                else:
                    init_values = [0.0] * num_inputs

                initial_data = struct.pack(self.fmt_send, *init_values)
                self.sock_send.sendto(initial_data, (self.host, send_port))
                if self.debug:
                    print(f"✅ 已发送初始数据包到端口{send_port}: {init_values}")
            except Exception as e:
                if self.debug:
                    print(f"⚠️ 发送初始数据包失败（Simulink可能未启动）: {e}")

    def call(self, inputs: List[float]) -> List[float]:
        """
        发送输入到Simulink，接收输出（替代matlab_engine.sim()）

        Args:
            inputs: 输入参数列表

        Returns:
            输出参数列表

        Raises:
            ValueError: 输入参数数量不匹配
            TimeoutError: 接收超时
        """
        # 验证输入数量
        if len(inputs) != self.num_inputs:
            raise ValueError(
                f"❌ 输入数量不匹配: 期望{self.num_inputs}, 实际{len(inputs)}"
            )

        start_time = time.time()

        # 打包并发送数据
        data = struct.pack(self.fmt_send, *inputs)
        self.sock_send.sendto(data, (self.host, self.send_port))

        # 接收响应
        try:
            resp, _ = self.sock_recv.recvfrom(1024)

            # 验证数据包大小
            expected_size = struct.calcsize(self.fmt_recv)
            if len(resp) != expected_size:
                raise ValueError(
                    f"❌ UDP数据包大小错误: 期望{expected_size}字节, 实际{len(resp)}字节"
                )

            # 解包数据
            outputs = list(struct.unpack(self.fmt_recv, resp))

            # 性能统计
            elapsed = time.time() - start_time
            self.call_count += 1
            self.total_time += elapsed

            if self.debug and self.call_count % 20 == 0:
                avg_time = self.total_time / self.call_count * 1000
                print(f"📊 UDP调用统计: 次数={self.call_count}, "
                      f"平均耗时={avg_time:.2f}ms")

            return outputs

        except socket.timeout:
            raise TimeoutError(
                f"❌ UDP接收超时({self.timeout}s)\n"
                f"   Simulink模型可能未运行或端口配置错误\n"
                f"   发送端口: {self.send_port}, 接收端口: {self.recv_port}\n"
                f"   请确认Simulink服务器已启动（run start_simulink_servers.m）"
            )

    def cleanup(self):
        """关闭socket连接"""
        try:
            self.sock_send.close()
            self.sock_recv.close()
            if self.debug:
                print(f"✅ UDP连接已关闭: {self.send_port}/{self.recv_port}")
        except Exception as e:
            print(f"⚠️ UDP关闭失败: {e}")

    def test_connection(self) -> bool:
        """
        测试UDP连接是否正常

        Returns:
            True: 连接正常, False: 连接失败
        """
        try:
            # 发送测试数据（全零）
            test_inputs = [0.0] * self.num_inputs
            outputs = self.call(test_inputs)

            print(f"✅ UDP连接测试成功: {self.send_port}/{self.recv_port}")
            print(f"   测试输出: {outputs}")
            return True

        except Exception as e:
            print(f"❌ UDP连接测试失败: {e}")
            return False


# 向后兼容性：如果项目中有代码使用旧名称
UDPClient = SimulinkUDPClient


if __name__ == '__main__':
    """简单的连接测试"""
    print("=== Simulink UDP接口测试 ===\n")

    # 测试决策模型连接
    print("1. 测试决策模型连接...")
    try:
        decision_client = SimulinkUDPClient(
            send_port=25000,
            recv_port=25001,
            num_inputs=4,
            num_outputs=5,
            timeout=0.5,
            debug=True
        )
        decision_client.test_connection()
        decision_client.cleanup()
    except Exception as e:
        print(f"   决策模型测试失败: {e}\n")

    # 测试控制模型连接
    print("\n2. 测试控制模型连接...")
    try:
        control_client = SimulinkUDPClient(
            send_port=26000,
            recv_port=26001,
            num_inputs=5,  # ⚠️ 注意：已从6个改为5个
            num_outputs=5,
            timeout=0.5,
            debug=True
        )
        control_client.test_connection()
        control_client.cleanup()
    except Exception as e:
        print(f"   控制模型测试失败: {e}\n")

    print("\n如果两个测试都成功，说明UDP通信系统就绪 ✅")
    print("如果失败，请确认：")
    print("  1. Simulink服务器已启动（run start_simulink_servers.m）")
    print("  2. 端口配置正确（25000/25001, 26000/26001）")
    print("  3. 防火墙未阻止本地UDP通信")
