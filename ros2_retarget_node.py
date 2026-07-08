#!/usr/bin/env python3
"""方案 B · 计算节点：keypoints → wuji_sdk.retargeting → /{side}_hand/joint_commands。

和客户方案 B 一致：用闭源 wuji_sdk.retargeting.RetargetSession（效果最好那条），
step() 输出已是固件关节序，原样 publish 到 ROS2 topic，下游（真机驱动 / 仿真）订阅。

关键点来源：
  --source replay  从录制 pkl 回放（无手套，演示用）
  --source glove   真手套（wuji_sdk hand_skeleton，drain 取最新帧）

工程要点（照方案文档）：一个 session 全程复用；drain 取最新；掉帧保持上一姿态不发零位；
ROS timer 定频（默认 120Hz）；QoS best-effort + depth1。
"""
import argparse
import pickle
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from wuji_sdk import Handedness, retargeting

JOINT_NAMES = [f"finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]


class RetargetNode(Node):
    def __init__(self, args):
        super().__init__(f"retarget_teleop_{args.side}")
        self._side = args.side
        hd = Handedness.Right if args.side == "right" else Handedness.Left
        model = (retargeting.HandModel.WujiHand2
                 if args.hand_model == "wuji_hand_2"
                 else retargeting.HandModel.WujiHand)
        self.get_logger().info("Building RetargetSession (few seconds)...")
        self._session = retargeting.RetargetSession.for_hand(model, side=hd)
        self._session.reset()

        self._pub = self.create_publisher(
            JointState, f"/{args.side}_hand/joint_commands", qos_profile_sensor_data)

        self._source = args.source
        self._last_kp = None
        if args.source == "replay":
            self._kps = self._load_replay(args.data, args.side)
            self._idx = 0
            self._loop = args.loop
            self.get_logger().info(f"replay {len(self._kps)} frames")
        else:
            self._setup_glove(args.glove_sn)

        self.create_timer(1.0 / args.rate, self._tick)
        self.get_logger().info(
            f"Ready: side={args.side} source={args.source} rate={args.rate}Hz "
            f"-> /{args.side}_hand/joint_commands")

    def _load_replay(self, path, side):
        frames = pickle.load(open(path, "rb"))
        key = f"{side}_fingers"
        kps = [np.asarray(f[key], dtype=np.float32)
               for f in frames if f.get(key) is not None]
        if not kps:
            raise SystemExit(f"no {key} in {path}")
        return kps

    def _setup_glove(self, sn):
        from wuji_sdk import SdkManager
        mgr = SdkManager.instance()
        dev = mgr.connect(sn=sn, device_name=sn) if sn else mgr.auto_connect(device_name="glove")
        self._glove_sub = dev.hand_skeleton().subscribe()

    def _next_keypoints(self):
        if self._source == "replay":
            if self._idx >= len(self._kps):
                if not self._loop:
                    return None
                self._idx = 0
            kp = self._kps[self._idx]
            self._idx += 1
            return kp
        # glove: drain 取最新帧
        frame = None
        while (f := self._glove_sub.recv()) is not None:
            frame = f
        if frame is None:
            return self._last_kp   # 掉帧保持上一姿态
        kp = np.asarray([j.pose.position for j in frame.joints], dtype=np.float32)
        return kp if kp.shape == (21, 3) else self._last_kp

    def _tick(self):
        kp = self._next_keypoints()
        if kp is None:
            return
        self._last_kp = kp
        qpos = self._session.step(kp)          # (20,) 固件序
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = [float(v) for v in qpos]
        self._pub.publish(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--hand-model", default="wuji_hand",
                    choices=["wuji_hand", "wuji_hand_2"])
    ap.add_argument("--source", default="replay", choices=["replay", "glove"])
    ap.add_argument("--data", default="/tmp/wuji-retargeting/example/data/avp1.pkl")
    ap.add_argument("--glove-sn", default="")
    ap.add_argument("--rate", type=float, default=120.0)
    ap.add_argument("--loop", action="store_true", default=True)
    args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = RetargetNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
