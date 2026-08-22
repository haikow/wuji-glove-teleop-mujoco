#!/usr/bin/env python3
"""回看 episode：把录下来的 action（或策略预测的 action）放回 MuJoCo 渲染成视频。

同一个入口既做人工回看，也做策略验证 —— `--policy` 时用 BC 模型在录下来的 obs 上
重新预测 action 并回放，同时打印和录制 action 的逐关节误差。

用法：
    MUJOCO_GL=egl ./venv312/bin/python tools/viz_episode.py data/episodes/ep_xxx
    MUJOCO_GL=egl ./venv312/bin/python tools/viz_episode.py data/episodes/ep_xxx \
        --policy data/models/bc.pt --out policy_replay.mp4
    ./venv312/bin/python tools/viz_episode.py data/episodes/ep_xxx --stats-only
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.episode_format import load_frames, load_meta, meta_path  # noqa: E402
from tools.export_dataset import _obs_vector  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_mjcf(hand_model, side, override=""):
    """按手型挑 MJCF：二代手要用 wuji_hand_description2/（本仓库只 vendor 了一代）。

    实测用一代 MJCF 去 clip 二代 retarget 输出，63.7% 的帧越界、最大 0.53 rad(30°)，
    回放和限位都会错。二代模型用 tools/fetch_hand2_description.sh 拉。
    """
    if override:
        return override
    if hand_model == "wuji_hand_2":
        p = os.path.join(ROOT, "wuji_hand_description2", "mjcf", "%s.xml" % side)
        if os.path.isfile(p):
            return p
        print("[warn] --hand-model wuji_hand_2 但找不到 wuji_hand_description2/，"
              "回落一代 MJCF —— 限位和回放会不准，跑 tools/fetch_hand2_description.sh 修复")
    return os.path.join(ROOT, "wuji_hand_description", "mjcf", "%s.xml" % side)



def setup_camera(m, mujoco):
    """按模型包围盒自适应相机 —— 一代/二代手的基座和尺度不同，硬编码 distance/lookat
    会让二代手渲成一个偏在角落的小点。用 m.stat.extent/center 自动取景。"""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation = 135, -20
    cam.distance = 2.2 * float(m.stat.extent)
    cam.lookat[:] = m.stat.center
    return cam


def _predict(policy_path, frames, obs_mode):
    """用训练好的 BC 模型在录下来的 obs 上重新预测 action。"""
    import torch
    from tools.train_bc import MLP

    ck = torch.load(policy_path, map_location="cpu", weights_only=False)
    mode = ck.get("obs_mode", obs_mode)
    net = MLP(ck["obs_dim"], ck["action_dim"], ck.get("hidden", 256),
              ck.get("layers", 2))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    om, os_, am, as_ = (np.asarray(ck[k], np.float32)
                        for k in ("obs_mean", "obs_std", "action_mean", "action_std"))
    X = np.asarray([_obs_vector(f, mode) for f in frames], np.float32)
    with torch.no_grad():
        Y = net(torch.from_numpy((X - om) / os_)).numpy()
    return Y * as_ + am, mode


def main():
    ap = argparse.ArgumentParser(description="episode 回看 / 策略回放")
    ap.add_argument("episode")
    ap.add_argument("--out", default="", help="输出 mp4；默认写到 episode 目录里")
    ap.add_argument("--source", default="action", choices=["action", "joint_angles"])
    ap.add_argument("--policy", default="", help="BC 模型 .pt，用它预测 action 来回放")
    ap.add_argument("--obs", default="both",
                    help="和导出/训练时用的 --obs 一致（默认从 checkpoint 里读）")
    ap.add_argument("--mjcf", default="", help="手动指定 MJCF；默认沿用 meta 里录制时那份")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="每 N 帧渲一帧")
    ap.add_argument("--stats-only", action="store_true", help="只算误差不渲染")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    args = ap.parse_args()

    meta = load_meta(args.episode) if os.path.isfile(meta_path(args.episode)) else {}
    side = meta.get("side") or "right"
    frames = [f for f in load_frames(args.episode) if isinstance(f.get(args.source), list)]
    if not frames:
        raise SystemExit("episode 里没有 %s 字段可回放" % args.source)

    recorded = np.asarray([f[args.source] for f in frames], np.float64)
    qpos_seq, label = recorded, "recorded " + args.source

    if args.policy:
        frames = [f for f in frames if _obs_vector(f, args.obs)]
        recorded = np.asarray([f[args.source] for f in frames], np.float64)
        pred, mode = _predict(args.policy, frames, args.obs)
        qpos_seq, label = pred, "policy(%s)" % os.path.basename(args.policy)
        err = np.abs(pred - recorded)
        print("策略 vs 录制 action（obs=%s，%d 帧）：" % (mode, len(frames)))
        print("  MAE  = %.5f rad (%.3f°)" % (err.mean(), np.degrees(err.mean())))
        print("  RMSE = %.5f rad" % np.sqrt(((pred - recorded) ** 2).mean()))
        print("  最差关节: idx=%d MAE=%.5f rad" % (err.mean(0).argmax(), err.mean(0).max()))

    if args.stats_only:
        return 0

    import mujoco
    import imageio

    # 录制时用的 MJCF 记在 meta 里，回放优先沿用它，保证限位一致
    mjcf = args.mjcf or meta.get("mjcf") or resolve_mjcf(
        meta.get("hand_model") or "wuji_hand", side)
    m = mujoco.MjModel.from_xml_path(mjcf)
    d = mujoco.MjData(m)
    jlo, jhi = m.jnt_range[:, 0].copy(), m.jnt_range[:, 1].copy()
    ren = mujoco.Renderer(m, args.h, args.w)
    cam = setup_camera(m, mujoco)

    out = args.out or os.path.join(args.episode, "replay.mp4")
    n = min(m.nq, qpos_seq.shape[1])
    with imageio.get_writer(out, fps=args.fps, macro_block_size=None) as w:
        for q in qpos_seq[::args.stride]:
            d.qpos[:n] = np.clip(q[:n], jlo[:n], jhi[:n])
            mujoco.mj_forward(m, d)
            ren.update_scene(d, cam)
            w.append_data(ren.render())
    ren.close()
    print("回放 %s → %s（%d 帧，%s）"
          % (os.path.basename(args.episode), out, len(qpos_seq[::args.stride]), label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
