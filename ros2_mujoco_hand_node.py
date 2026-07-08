#!/usr/bin/env python3
"""方案 B · 输出 sink（仿真）：订阅 /{side}_hand/joint_commands → 驱动 MuJoCo 手模型。

这就是方案 B 里"输出端指向仿真"的那条：命令流走 ROS2 topic，仿真手订阅即动；
换成真机时把本节点换成 wujihandros2/自研驱动即可，计算节点完全不用改。

下游不重排关节：position[20] 已是固件序，直接进 data.ctrl（按 ctrlrange 裁剪）。

模式：
  --mode view   开窗实时看（需 DISPLAY）
  --mode video  离屏录制，收到 --frames 帧后写出 mp4 退出（无显示器/CI 可用）
"""
import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

HERE = Path(__file__).resolve().parent


class MujocoHandNode(Node):
    def __init__(self, args):
        super().__init__(f"mujoco_hand_{args.side}")
        mjcf = args.model or str(HERE / "wuji_hand_description" / "mjcf" / f"{args.side}.xml")
        self.m = mujoco.MjModel.from_xml_path(mjcf)
        self.d = mujoco.MjData(self.m)
        self._lo = self.m.actuator_ctrlrange[:, 0]
        self._hi = self.m.actuator_ctrlrange[:, 1]
        self._ctrl = np.zeros(self.m.nu)
        self._lock = threading.Lock()
        self._got = 0

        self.create_subscription(
            JointState, f"/{args.side}_hand/joint_commands",
            self._on_cmd, qos_profile_sensor_data)
        self.get_logger().info(
            f"subscribed /{args.side}_hand/joint_commands, MJCF nu={self.m.nu}")

    def _on_cmd(self, msg: JointState):
        q = np.asarray(msg.position, dtype=np.float64)
        if q.shape[0] != self.m.nu:
            return
        with self._lock:
            self._ctrl = np.clip(q, self._lo, self._hi)
            self._got += 1

    def step_ctrl(self, substeps=2):
        with self._lock:
            self.d.ctrl[:] = self._ctrl
        for _ in range(substeps):
            mujoco.mj_step(self.m, self.d)


def _cam(obj):
    obj.azimuth, obj.elevation, obj.distance = 180, -20, 0.42
    obj.lookat[:] = [0, 0, 0.05]


def run_view(node):
    import mujoco.viewer
    ex = rclpy.executors.SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    with mujoco.viewer.launch_passive(node.m, node.d) as v:
        _cam(v.cam)
        import time
        while v.is_running() and rclpy.ok():
            node.step_ctrl()
            v.sync()
            time.sleep(node.m.opt.timestep * 2)


def run_video(node, out, frames, fps):
    import imageio
    ren = mujoco.Renderer(node.m, height=720, width=960)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); _cam(cam)
    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    written = 0
    # spin ROS in bg; sample the latest ctrl at fps into the video
    ex = rclpy.executors.SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    import time
    # wait for first command
    t0 = time.time()
    while node._got == 0 and time.time() - t0 < 10:
        time.sleep(0.05)
    node.get_logger().info(f"first cmd after {time.time()-t0:.2f}s, recording {frames} frames")
    for _ in range(frames):
        node.step_ctrl()
        ren.update_scene(node.d, cam)
        writer.append_data(ren.render())
        written += 1
        time.sleep(1.0 / fps)
    writer.close()
    ren.close()
    node.get_logger().info(f"[video] wrote {out} ({written} frames), total cmds recv={node._got}")
    # 硬退出，跳过 GL/rclpy 多线程 teardown 竞争（否则退出时偶发 core dump）
    import os
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", default="view", choices=["view", "video"])
    ap.add_argument("--out", default="teleop_ros2.mp4")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = MujocoHandNode(args)
    try:
        if args.mode == "view":
            run_view(node)
        else:
            run_video(node, args.out, args.frames, args.fps)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
