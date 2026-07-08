#!/usr/bin/env python3
"""方案 B 效果预览：录制手部关键点 → wuji_sdk.retargeting → MuJoCo 手模型。

用的就是"效果最好"的闭源 SDK retargeting（wuji_sdk.retargeting.RetargetSession），
和 1.teleop_real.py / 客户方案 B 的算法完全一致；这里只是把输出喂给 MuJoCo 仿真手
而不是真机，方便直接看遥操效果。

无需手套：从录制的 MediaPipe 关键点 (21,3) 回放。

用法：
    # 离屏渲染出 mp4（无显示器也能跑，产出可分享）
    python mujoco_teleop_replay.py --mode video --out teleop.mp4 --frames 900

    # 本机开窗实时看（需要 DISPLAY）
    python mujoco_teleop_replay.py --mode view
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import mujoco

from wuji_sdk import Handedness, retargeting

HERE = Path(__file__).resolve().parent


def load_keypoints(data_path: str, side: str):
    """回放数据 -> list[(21,3) float32]，取指定手且非空的帧。"""
    frames = pickle.load(open(data_path, "rb"))
    key = f"{side}_fingers"
    kps = [
        np.asarray(f[key], dtype=np.float32)
        for f in frames
        if f.get(key) is not None
    ]
    if not kps:
        raise SystemExit(f"回放数据里没有 {key} 帧：{data_path}")
    return kps


def retarget_all(kps, hand_model, side):
    """整段回放先离线跑一遍 retargeting，得到每帧 (20,) qpos（固件关节序）。"""
    model = (
        retargeting.HandModel.WujiHand2
        if hand_model == "wuji_hand_2"
        else retargeting.HandModel.WujiHand
    )
    hd = Handedness.Right if side == "right" else Handedness.Left
    sess = retargeting.RetargetSession.for_hand(model, side=hd)
    sess.reset()
    # session 全程复用（内部持 warm-start + 低通状态），逐帧 step
    return np.asarray([sess.step(k) for k in kps], dtype=np.float64)


def clip_to_ctrlrange(m, q):
    lo = m.actuator_ctrlrange[:, 0]
    hi = m.actuator_ctrlrange[:, 1]
    return np.clip(q, lo, hi)


def run_video(m, d, qpos, out, fps, substeps):
    import imageio
    W, H = 960, 720
    ren = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = 180, -20, 0.42
    cam.lookat[:] = [0, 0, 0.05]
    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    for i in range(len(qpos)):
        d.ctrl[:] = qpos[i]
        for _ in range(substeps):
            mujoco.mj_step(m, d)
        ren.update_scene(d, cam)
        writer.append_data(ren.render())
    writer.close()
    print(f"[video] 写出 {out}  帧数={len(qpos)}  fps={fps}")


def run_view(m, d, qpos, fps, substeps, loop):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 180, -20, 0.42
        viewer.cam.lookat[:] = [0, 0, 0.05]
        dt = 1.0 / fps
        i = 0
        n = len(qpos)
        while viewer.is_running():
            d.ctrl[:] = qpos[i % n]
            for _ in range(substeps):
                mujoco.mj_step(m, d)
            viewer.sync()
            i += 1
            if not loop and i >= n:
                break
            time.sleep(dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--hand-model", default="wuji_hand",
                    choices=["wuji_hand", "wuji_hand_2"])
    ap.add_argument("--data", default="/tmp/wuji-retargeting/example/data/avp1.pkl")
    ap.add_argument("--model", default=None, help="MJCF 路径，默认按 side 取 wuji_hand_description")
    ap.add_argument("--mode", default="video", choices=["video", "view"])
    ap.add_argument("--out", default="teleop.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=0, help="只取前 N 帧（0=全部）")
    ap.add_argument("--substeps", type=int, default=8, help="每帧 mj_step 子步数")
    ap.add_argument("--loop", action="store_true", help="view 模式循环回放")
    args = ap.parse_args()

    mjcf = args.model or str(HERE / "wuji_hand_description" / "mjcf" / f"{args.side}.xml")
    m = mujoco.MjModel.from_xml_path(mjcf)
    d = mujoco.MjData(m)
    assert m.nu == 20, f"MJCF 执行器数 {m.nu} != 20"

    kps = load_keypoints(args.data, args.side)
    if args.frames:
        kps = kps[: args.frames]
    print(f"[retarget] {len(kps)} 帧  side={args.side}  model={args.hand_model}")
    qpos = retarget_all(kps, args.hand_model, args.side)
    qpos = clip_to_ctrlrange(m, qpos)

    if args.mode == "video":
        run_video(m, d, qpos, args.out, args.fps, args.substeps)
    else:
        run_view(m, d, qpos, args.fps, args.substeps, args.loop)


if __name__ == "__main__":
    main()
