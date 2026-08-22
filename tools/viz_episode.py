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


# MediaPipe 手骨架连线（同 glove_teleop_live.SK_CONN）
SK_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
           (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
           (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]


def _mesh_geoms(m, mujoco):
    """MJCF 里可见的 mesh geom → [(geom_id, verts, faces)]，顶点是 geom 局部系。

    每个 body 通常有 visual+collision 两个同名 mesh geom，按 mesh id 去重只留一份。
    用 geom（而不是 body）当实体：MuJoCo 的 d.geom_xpos/xmat 已经是世界位姿，
    不用再手动复合 body→geom 的偏移。
    """
    import numpy as np
    out, seen = [], set()
    for g in range(m.ngeom):
        mid = int(m.geom_dataid[g])
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or mid < 0 or mid in seen:
            continue
        seen.add(mid)
        v0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        f0, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        verts = np.asarray(m.mesh_vert[v0:v0 + nv], np.float32).reshape(-1, 3)
        faces = np.asarray(m.mesh_face[f0:f0 + nf], np.uint32).reshape(-1, 3)
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) or ("mesh%d" % mid)
        out.append((g, name, verts, faces))
    return out


def _log_hand_static(rr, root, geoms, color):
    """静态 log 一次网格；之后每帧只更新 Transform3D，rrd 体积不会随帧数爆。"""
    for _g, name, verts, faces in geoms:
        rr.log("%s/%s" % (root, name),
               rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                         albedo_factor=color),
               static=True)


def _log_hand_pose(rr, root, geoms, m, d, qpos, mujoco, nq):
    """置关节角 → mj_forward → 把每个 geom 的世界位姿写成 Transform3D。"""
    import numpy as np
    d.qpos[:nq] = np.clip(qpos[:nq], m.jnt_range[:nq, 0], m.jnt_range[:nq, 1])
    mujoco.mj_forward(m, d)
    for g, name, _v, _f in geoms:
        rr.log("%s/%s" % (root, name),
               rr.Transform3D(translation=d.geom_xpos[g],
                              mat3x3=d.geom_xmat[g].reshape(3, 3)))


def _blueprint(rrb, has_actual, has_policy):
    """3D 视图占主位，曲线收进右侧标签页 —— 否则 80 条标量会把 3D 挤没。"""
    tabs = [rrb.TimeSeriesView(origin="/action", name="指令 action")]
    if has_actual:
        tabs.append(rrb.TimeSeriesView(origin="/track_err", name="跟踪误差"))
        tabs.append(rrb.TimeSeriesView(origin="/hand_state", name="真机实测"))
    if has_policy:
        tabs.append(rrb.TimeSeriesView(origin="/policy", name="策略预测"))
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="手 + 手套骨架"),
            rrb.Tabs(*tabs),
            column_shares=[3, 2]),
        collapse_panels=True)


def log_rerun(ep_dir, frames, out, joint_names=None, pred=None, stride=1,
              mjcf=None, meta=None):
    """把 episode 写成 .rrd。

    3D 视图里有三层，同一坐标系下可直接比对：
      - `/world/robot_cmd`   指令位姿下的机器人手（**真网格**，不是点云）
      - `/world/robot_real`  真机 joint_states 实测位姿（有真机数据时）
      - `/world/glove`       手套 21 点人手骨架（对齐到腕部）
    加上逐关节 指令/实测/误差/策略 时间序列，外带 blueprint 让 3D 占主视图。

    落盘而不是开窗：这台机器常跑无头 EGL。拿到 .rrd 后本地 `rerun <file>.rrd` 打开。
    """
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb

    meta = meta or {}
    names = joint_names or ["j%d" % i for i in range(20)]
    has_actual = any(isinstance(f.get("hand_state"), list) for f in frames)
    has_policy = pred is not None

    rr.init("wuji_episode_%s" % os.path.basename(os.path.normpath(ep_dir)))
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    geoms = m = d = None
    if mjcf:
        import mujoco
        m = mujoco.MjModel.from_xml_path(mjcf)
        d = mujoco.MjData(m)
        geoms = _mesh_geoms(m, mujoco)
        _log_hand_static(rr, "/world/robot_cmd", geoms, [200, 200, 210, 255])
        if has_actual:
            _log_hand_static(rr, "/world/robot_real", geoms, [90, 170, 255, 140])
        print("  rerun: 机器人手网格 %d 块（%s）" % (len(geoms), os.path.basename(mjcf)))

    t0 = frames[0].get("t_dev_us") or 0
    for i, f in enumerate(frames[::stride]):
        k = i * stride
        rr.set_time("t", duration=((f.get("t_dev_us") or t0) - t0) / 1e6)

        act = f.get("action")
        hs = f.get("hand_state")
        if geoms is not None:
            if act:
                _log_hand_pose(rr, "/world/robot_cmd", geoms, m, d,
                               np.asarray(act, float), mujoco, m.nq)
            if has_actual and hs and all(v is not None for v in hs):
                _log_hand_pose(rr, "/world/robot_real", geoms, m, d,
                               np.asarray(hs, float), mujoco, m.nq)

        sk = f.get("skeleton")
        if sk:
            pts = np.asarray(sk, np.float32)
            conf = f.get("confidence") or [1.0] * len(pts)
            # 置信度低的点染红，一眼看出哪根手指跟丢了
            cols = np.array([[255, 140, 0] if c >= 0.3 else [255, 40, 40]
                             for c in conf], np.uint8)
            rr.log("/world/glove/joints", rr.Points3D(pts, radii=0.005, colors=cols))
            rr.log("/world/glove/bones",
                   rr.LineStrips3D([[pts[a], pts[b]] for a, b in SK_CONN],
                                   radii=0.002, colors=[255, 170, 60]))

        for j, nm in enumerate(names[:20]):
            if act and j < len(act):
                rr.log("/action/%s" % nm, rr.Scalars(act[j]))
            if hs and j < len(hs) and hs[j] is not None:
                rr.log("/hand_state/%s" % nm, rr.Scalars(hs[j]))
                if act and j < len(act):
                    rr.log("/track_err/%s" % nm, rr.Scalars(abs(hs[j] - act[j])))
            if has_policy and j < pred.shape[1]:
                rr.log("/policy/%s" % nm, rr.Scalars(float(pred[k, j])))

    rr.save(out, default_blueprint=_blueprint(rrb, has_actual, has_policy))
    return out


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
    ap.add_argument("--rerun", action="store_true",
                    help="写 .rrd（3D 骨架 + 指令/实测/误差时间序列），本地 rerun 打开")
    ap.add_argument("--rerun-stride", type=int, default=2, help="rerun 每 N 帧记一次")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    args = ap.parse_args()

    meta = load_meta(args.episode) if os.path.isfile(meta_path(args.episode)) else {}
    side = meta.get("side") or "right"
    frames = [f for f in load_frames(args.episode) if isinstance(f.get(args.source), list)]
    if not frames:
        raise SystemExit("episode 里没有 %s 字段可回放" % args.source)

    # 录制时用的 MJCF 记在 meta 里，回放/rerun 都沿用它，保证限位与几何一致
    mjcf_path = args.mjcf or meta.get("mjcf") or resolve_mjcf(
        meta.get("hand_model") or "wuji_hand", side)

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

    if args.rerun:
        out = args.out or os.path.join(args.episode, "episode.rrd")
        if not out.endswith(".rrd"):
            out = os.path.splitext(out)[0] + ".rrd"
        jn = (meta.get("action_space") or {}).get("joint_names")
        log_rerun(args.episode, frames, out, jn,
                  pred if args.policy else None, args.rerun_stride,
                  mjcf=mjcf_path, meta=meta)
        print("rerun → %s（本地打开： rerun %s）" % (out, out))
        return 0

    if args.stats_only:
        return 0

    import mujoco
    import imageio

    m = mujoco.MjModel.from_xml_path(mjcf_path)
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
