#!/usr/bin/env python3
"""方案 B · 输出 sink（真机·二代手）：订阅 /{side}_hand/joint_commands → 用 wuji_sdk
驱动真 Wuji Hand 2；同时把实测关节角回发到 /{side}_hand/joint_states。

二代手目前没有官方 ROS2 包（wuji-ros2 仍在设计阶段；wujihandros2 只支持一代手），
所以这里用一个很薄的 rclpy 节点把“非 ROS2 的手”封装成标准 ROS2 接口：
  订阅 sensor_msgs/JointState.position[20]（固件序，与 ros2_retarget_node 一致）
    → hand.joint_command().publish().send([JointCommand(pos, vel, 0), ...])
  订阅 hand.joint_states()（SDK 流）→ 发 sensor_msgs/JointState 回 ROS2。

与 ros2_mujoco_hand_node.py 可互换：把仿真 sink 换成本节点即可上真机，计算节点不动。

下游不重排关节：position[20] 已是固件序，直接下发。若上游 JointState 带 velocity[20]，
作为 MIT 速度前馈；否则前馈 0。

安全：会使能真手并让它跟随命令实时运动。首帧做一次“从当前实测位置平滑过渡到第一条
命令”的短插值，避免上电瞬间猛弹（同 2.publish.py 平滑回零的思路）。Ctrl+C 自动 disable。

依赖：pip install wuji-sdk（cp312 = ROS2 Humble/Kilted 的 ABI，一套 env 通吃）。
用法：
    ros2 run ...            # 或直接：
    python ros2_realhand_node.py --side right
    python ros2_realhand_node.py --side right --hand-sn <SN> --kp 3.0 --kd 0.05 --effort 1.5
"""
import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from wuji_sdk import DeviceType, JointCommand, SdkManager, WujiHand2

TOTAL_JOINTS = 20
JOINT_NAMES = [f"finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
RAMP_SECONDS = 1.0   # 首帧从当前位置平滑过渡到第一条命令的时长


class RealHandNode(Node):
    def __init__(self, args):
        super().__init__(f"wuji_hand2_{args.side}")
        self.side = args.side

        # ---- 连接 + 使能真二代手 ----
        mgr = SdkManager.instance()
        self.mgr = mgr
        hands = [d for d in mgr.scan() if d.device_type == DeviceType.WujiHand2
                 and (not args.hand_sn or args.hand_sn in str(d.sn))]
        if not hands:
            raise RuntimeError("未找到二代手 Wuji Hand 2（检查上电/网络/--hand-sn）")
        self.hand: WujiHand2 = mgr.connect(sn=hands[0].sn, device_name="wuji_hand_2")
        n_online = self.hand.online_joints_count().get()
        self.get_logger().info(f"connected {self.hand.serial_number} ({n_online} joints online)")
        if n_online == 0:
            raise RuntimeError("无在线关节")

        self.hand.effort_limit().set(args.effort)
        self.hand.mit_params().set((args.kp, args.kd))
        self.hand.enable()
        if not self._wait_enabled():
            raise RuntimeError("enable 超时")
        self.get_logger().info("all motors enabled")

        self._pub_cmd = self.hand.joint_command().publish()
        self._homed = False

        # ---- ROS2 接口 ----
        self._sub = self.create_subscription(
            JointState, f"/{args.side}_hand/joint_commands",
            self._on_cmd, qos_profile_sensor_data)
        self._pub_state = self.create_publisher(
            JointState, f"/{args.side}_hand/joint_states", qos_profile_sensor_data)
        # 回发实测关节角：定时器排空 joint_states 流到最新帧再发
        self._state_sub = self.hand.joint_states().subscribe()
        self.create_timer(1.0 / max(1, args.state_rate), self._pub_joint_states)
        self.get_logger().info(
            f"subscribed /{args.side}_hand/joint_commands ; publishing /{args.side}_hand/joint_states")

    # ---- helpers ----
    def _wait_enabled(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        sub = self.hand.joint_diagnostics().subscribe()
        try:
            while time.monotonic() < deadline:
                time.sleep(0.2)
                f = sub.recv()
                if f and f.joints and all(e.status_word.ext_state == 2 for e in f.joints):
                    return True
            return False
        finally:
            sub.close()

    def _read_current(self, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            f = self._state_sub.recv()
            if f is not None and f.joints:
                return [float(j.position) if j.position is not None else 0.0 for j in f.joints]
            time.sleep(0.01)
        return [0.0] * TOTAL_JOINTS

    def _send(self, pos, vel):
        self._pub_cmd.send([JointCommand(p, v, 0.0) for p, v in zip(pos, vel)])

    # ---- ROS2 callbacks ----
    def _on_cmd(self, msg: JointState):
        pos = list(msg.position[:TOTAL_JOINTS])
        if len(pos) < TOTAL_JOINTS:
            return
        vel = list(msg.velocity[:TOTAL_JOINTS]) if len(msg.velocity) >= TOTAL_JOINTS \
            else [0.0] * TOTAL_JOINTS

        if not self._homed:
            # 首帧：从当前实测位置平滑过渡到第一条命令，避免上电猛弹
            start = self._read_current()
            steps = max(1, int(RAMP_SECONDS * 200))
            for i in range(1, steps + 1):
                a = i / steps
                q = [s + a * (p - s) for s, p in zip(start, pos)]
                self._send(q, [0.0] * TOTAL_JOINTS)
                time.sleep(1.0 / 200)
            self._homed = True
            self.get_logger().info("initial smooth-home done; passing commands through")
            return

        self._send(pos, vel)

    def _pub_joint_states(self):
        latest = None
        while True:
            f = self._state_sub.recv()
            if f is None:
                break
            latest = f
        if latest is None or not latest.joints:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = [float(j.position) if j.position is not None else 0.0
                        for j in latest.joints]
        self._pub_state.publish(msg)

    def shutdown(self):
        with __import__("contextlib").suppress(Exception):
            self.hand.disable()
        with __import__("contextlib").suppress(Exception):
            self.mgr.disconnect_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", help="left/right（决定 topic 前缀）")
    ap.add_argument("--hand-sn", default="", help="二代手 SN；总线多手时指定")
    ap.add_argument("--kp", type=float, default=3.0, help="MIT kp")
    ap.add_argument("--kd", type=float, default=0.05, help="MIT kd")
    ap.add_argument("--effort", type=float, default=1.5, help="电流上限 A")
    ap.add_argument("--state-rate", type=int, default=100, help="joint_states 回发频率 Hz")
    args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = None
    try:
        node = RealHandNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
