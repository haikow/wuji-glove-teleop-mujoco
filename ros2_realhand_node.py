#!/usr/bin/env python3
"""方案 B · 输出 sink（真机·二代手）：订阅 <prefix>joint_commands → 用 wuji_sdk 驱动真
Wuji Hand 2（**以太网 zenoh/UDP**，非 USB）；回发 <prefix>joint_states，可选发
<prefix>hand_diagnostics。可作为 wujihandros2（一代手 USB 驱动）的**二代手以太网即插替换**。

背景：二代手没有官方 ROS2 包（wuji-ros2 仍在设计阶段），wujihandros2 的驱动是
`wujihandcpp::device::Hand(serial)` = **USB / 一代手**，没有 UDP/以太网 transport，连不了
走以太网的二代手。本节点用 `wuji_sdk`（scan/connect 走以太网 zenoh/UDP）补上这块，且沿用
wujihandros2 的话题契约：
  订阅 <prefix>joint_commands（sensor_msgs/JointState, position[20] 固件序，
    带 velocity[20] 则作 MIT 速度前馈）→ hand.joint_command().publish().send(...)
  发 <prefix>joint_states（sensor_msgs/JointState）
  可选发 <prefix>hand_diagnostics（wujihand_msgs/HandDiagnostics，--diagnostics）

话题前缀（--prefix）：
  - 默认 `/{side}_hand/`（与本仓 ros2_retarget_node 一致）。
  - 顶替 wujihandros2 单手默认：`--prefix ""`（相对名 joint_commands/…）。
  - 顶替 wujihandros2 多手：`--prefix /hand_right/`（按手性绝对名）。
  - 或直接用 ROS2 remap：`--ros-args -r <old>:=<new>`。

安全：会使能真手并让它跟随命令实时运动。首帧从当前实测位置平滑过渡到第一条命令，避免上电
猛弹。Ctrl+C 自动 disable。

环境（易踩坑）：rclpy 与所用 Python 版本必须匹配（如 ROS2 Kilted = python3.12）；用
python3.12 跑并把装了 wuji_sdk 的 site-packages 挂到 PYTHONPATH。--diagnostics 需要
wuji-hand-teleop / wujihandros2 环境里的 wujihand_msgs 包（否则自动跳过该话题）。

用法：
    python3.12 ros2_realhand_node.py --side right --hand-sn <二代手SN>
    python3.12 ros2_realhand_node.py --side right --prefix "" --diagnostics   # 顶替 wujihandros2
"""
import argparse
import contextlib
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from wuji_sdk import DeviceType, JointCommand, SdkManager, WujiHand2

TOTAL_JOINTS = 20
JOINT_NAMES = [f"finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
RAMP_SECONDS = 1.0   # 首帧从当前位置平滑过渡到第一条命令的时长


def _drain_latest(sub):
    """排空订阅流，返回最新一帧（无则 None）。"""
    latest = None
    while True:
        f = sub.recv()
        if f is None:
            break
        latest = f
    return latest


class RealHandNode(Node):
    def __init__(self, args):
        super().__init__(f"wuji_hand2_{args.side}")
        self.side = args.side
        # 话题前缀：默认 /{side}_hand/；可 --prefix 覆盖（"" = 相对名，顶替 wujihandros2）
        self.prefix = args.prefix if args.prefix is not None else f"/{args.side}_hand/"

        # ---- 连接（以太网 zenoh/UDP）+ 使能真二代手 ----
        mgr = SdkManager.instance()
        self.mgr = mgr
        hands = [d for d in mgr.scan() if d.device_type == DeviceType.WujiHand2
                 and (not args.hand_sn or args.hand_sn in str(d.sn))]
        if not hands:
            raise RuntimeError("未找到二代手 Wuji Hand 2（检查上电/以太网/--hand-sn）")
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
            JointState, self.prefix + "joint_commands", self._on_cmd, qos_profile_sensor_data)
        self._pub_state = self.create_publisher(
            JointState, self.prefix + "joint_states", qos_profile_sensor_data)
        self._state_sub = self.hand.joint_states().subscribe()
        self.create_timer(1.0 / max(1, args.state_rate), self._pub_joint_states)

        # ---- 可选：hand_diagnostics（wujihand_msgs/HandDiagnostics）----
        self._diag_pub = None
        self._diag_sub = None
        if args.diagnostics:
            try:
                from wujihand_msgs.msg import HandDiagnostics
                self._HandDiagnostics = HandDiagnostics
                self._diag_pub = self.create_publisher(
                    HandDiagnostics, self.prefix + "hand_diagnostics", 10)
                self._diag_sub = self.hand.joint_diagnostics().subscribe()
                self.create_timer(1.0 / max(1, args.diag_rate), self._pub_diag)
            except ImportError:
                self.get_logger().warn(
                    "wujihand_msgs 不可用，跳过 hand_diagnostics（需在 wuji-hand-teleop/"
                    "wujihandros2 环境里；否则去掉 --diagnostics）")

        self.get_logger().info(
            f"prefix='{self.prefix}'  sub {self.prefix}joint_commands  "
            f"pub {self.prefix}joint_states"
            + (f" + {self.prefix}hand_diagnostics" if self._diag_pub else ""))

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
        latest = _drain_latest(self._state_sub)
        if latest is None or not latest.joints:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = [float(j.position) if j.position is not None else 0.0
                        for j in latest.joints]
        self._pub_state.publish(msg)

    def _pub_diag(self):
        latest = _drain_latest(self._diag_sub)
        if latest is None or not latest.joints:
            return
        js = latest.joints
        # 固定长度 20 数组（HandDiagnostics.msg 是 float32[20] 等定长）
        temps = [0.0] * TOTAL_JOINTS
        vbus = [0.0] * TOTAL_JOINTS
        errs = [0] * TOTAL_JOINTS
        en = [False] * TOTAL_JOINTS
        for i, j in enumerate(js[:TOTAL_JOINTS]):
            temps[i] = float(getattr(j, "mcu_temp_c_fb", 0.0) or 0.0)
            vbus[i] = float(getattr(j, "vbus_v_fb", 0.0) or 0.0)
            errs[i] = int(getattr(j, "error_code_current", 0) or 0)
            en[i] = (j.status_word.ext_state == 2)
        eff = self.hand.effort_limit().get()
        effort = [float(e) if e is not None else 0.0 for e in (eff or [])][:TOTAL_JOINTS]
        effort += [0.0] * (TOTAL_JOINTS - len(effort))

        m = self._HandDiagnostics()
        m.header.stamp = self.get_clock().now().to_msg()
        m.handedness = self.side
        m.system_temperature = max(temps) if temps else 0.0
        m.input_voltage = (sum(vbus) / len(vbus)) if vbus else 0.0
        m.joint_temperatures = temps
        m.error_codes = errs
        m.enabled = en
        m.effort_limits = effort
        self._diag_pub.publish(m)

    def shutdown(self):
        with contextlib.suppress(Exception):
            self.hand.disable()
        with contextlib.suppress(Exception):
            self.mgr.disconnect_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", help="left/right（决定默认前缀/句柄名）")
    ap.add_argument("--prefix", default=None,
                    help="话题前缀；默认 /{side}_hand/。顶替 wujihandros2 单手用 \"\"，多手用 /hand_right/")
    ap.add_argument("--hand-sn", default="", help="二代手 SN；总线多手时指定")
    ap.add_argument("--kp", type=float, default=3.0, help="MIT kp")
    ap.add_argument("--kd", type=float, default=0.05, help="MIT kd")
    ap.add_argument("--effort", type=float, default=1.5, help="电流上限 A")
    ap.add_argument("--state-rate", type=int, default=100, help="joint_states 回发频率 Hz")
    ap.add_argument("--diagnostics", action="store_true",
                    help="发 hand_diagnostics（wujihand_msgs/HandDiagnostics，需该消息包在环境里）")
    ap.add_argument("--diag-rate", type=int, default=10, help="hand_diagnostics 频率 Hz")
    args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = None
    try:
        node = RealHandNode(args)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # rclpy 在 SIGINT 时抛 ExternalShutdownException（不是 KeyboardInterrupt）
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
