#!/usr/bin/env python3
"""把 SDK 录的 MCAP 直接渲染成 Rerun `.rrd`（官方示例只给 MCAP，没有可视化）。

吃 `wuji_sdk.TopicRecorder` 录出来的任意 MCAP（如官方
`examples/python/wuji_glove/2.recording.py` 的 tactile + emf_poses + hand_skeleton），
不需要 episode 目录、不需要连设备。

渲染内容
  - `/world/skeleton`   21 关节点 + 骨架连线，**并用每个关节的四元数画出朝向**
                        （实测 19/21 关节带真实旋转，只有 wrist 和 thumb_cmc 是单位四元数）
  - `/world/emf`        5 个指尖 EMF 接收线圈的 6-DoF 位姿
  - `/tactile`          744 taxel → 24×31 图像（-1 = 无效/屏蔽）
  - `/world/robot`      `--retarget` 时：把骨架喂 RetargetSession 得到 20 维关节角，
                        再用 MJCF 真网格渲染机器人手（离线复算，同样不连设备）

⚠️ 坐标系：`hand_skeleton` 在 `r_wrist`、`emf_poses` 在 `r_hand_emf_tx`，两者不同。
本工具各自独立成组渲染，**没有把它们对齐到同一世界系** —— 那需要发射器到相机/世界的
外参，录制数据里没有。要接相机做同步渲染，得自己补这层标定。

用法：
    ./venv312/bin/python tools/viz_mcap.py /tmp/offrec/official.mcap
    ./venv312/bin/python tools/viz_mcap.py rec.mcap --retarget --side right \\
        --hand-model wuji_hand_2 -o rec.rrd
    ./venv312/bin/rerun rec.rrd
"""
import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.viz_episode import (  # noqa: E402
    SK_CONN, _log_hand_pose, _log_hand_static, _mesh_geoms, resolve_mjcf,
)

TACTILE_ROWS, TACTILE_COLS = 24, 31

# 蓝→青→绿→黄→红 的分段线性色标。不用 matplotlib（没装），5 个锚点足够读数。
_CMAP = np.array([[12, 20, 70], [0, 140, 200], [0, 200, 120],
                  [240, 220, 40], [220, 40, 30]], np.float32)
_INVALID_RGB = np.array([45, 45, 50], np.uint8)     # -1 = 无效/屏蔽 taxel


def tactile_rgb(data, vmax=1.0, rows=TACTILE_ROWS, cols=TACTILE_COLS):
    """744 个 taxel → HxWx3 uint8 热力图。

    单通道 float 直接 log 成 rr.Image 会被按灰度画，0~0.9 的值几乎全黑、看不出接触，
    所以这里自己上色标。-1 的无效 taxel 单独染深灰，不和"无接触"混为一谈。
    """
    v = np.asarray(data, np.float32).reshape(rows, cols)
    invalid = v < 0
    t = np.clip(v / max(vmax, 1e-6), 0.0, 1.0)
    idx = t * (len(_CMAP) - 1)
    lo = np.clip(np.floor(idx).astype(int), 0, len(_CMAP) - 2)
    frac = (idx - lo)[..., None]
    rgb = (_CMAP[lo] * (1 - frac) + _CMAP[lo + 1] * frac).astype(np.uint8)
    rgb[invalid] = _INVALID_RGB
    return rgb


def read_mcap(path, topics=("hand_skeleton", "emf_poses", "tactile",
                            "hand_joint_angles", "joint_states")):
    """MCAP → {topic: [(t_dev_us, payload_dict)]}，按设备时间戳排序。"""
    from mcap.reader import make_reader

    out = {t: [] for t in topics}
    with open(path, "rb") as f:
        for _sch, chan, msg in make_reader(f).iter_messages():
            t = chan.topic.rsplit("/", 1)[-1]
            if t not in out:
                continue
            try:
                d = json.loads(msg.data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            ts = (d.get("header") or {}).get("timestamp_us")
            out[t].append((ts if ts is not None else msg.log_time // 1000, d))
    for t in out:
        out[t].sort(key=lambda x: x[0])
    return {t: v for t, v in out.items() if v}


def _pose_arrays(poses):
    """[{pose:{position,orientation},confidence}] → (xyz Nx3, quat Nx4 xyzw, conf N)"""
    xyz, quat, conf = [], [], []
    for p in poses:
        pose = p.get("pose", p)
        pos = pose["position"]
        o = pose["orientation"]
        xyz.append(pos if isinstance(pos, list) else [pos[k] for k in "xyz"])
        quat.append([o["x"], o["y"], o["z"], o["w"]])
        conf.append(p.get("confidence", 1.0))
    return (np.asarray(xyz, np.float32), np.asarray(quat, np.float32),
            np.asarray(conf, np.float32))


def _blueprint(rrb, has_tactile, has_emf, has_robot):
    """3D 占主位；触觉图**单独给一个常驻视图**，不塞进标签页里（塞进去不选中就看不见）。"""
    tabs = [rrb.TimeSeriesView(origin="/conf", name="关节置信度")]
    if has_tactile:
        tabs.insert(0, rrb.TimeSeriesView(origin="/tactile_stat", name="触觉峰值/接触数"))
    if has_robot:
        tabs.append(rrb.TimeSeriesView(origin="/retarget", name="retarget 关节角"))
    right = [rrb.Tabs(*tabs)]
    if has_tactile:
        right.insert(0, rrb.Spatial2DView(origin="/tactile", name="触觉热力图 24×31"))
    return rrb.Blueprint(
        rrb.Horizontal(rrb.Spatial3DView(origin="/world", name="手骨架 / EMF / 机器人手"),
                       rrb.Vertical(*right, row_shares=[2, 1]) if has_tactile else right[0],
                       column_shares=[3, 2]),
        collapse_panels=True)


def main():
    ap = argparse.ArgumentParser(description="SDK 录的 MCAP → Rerun .rrd")
    ap.add_argument("mcap")
    ap.add_argument("-o", "--out", default="", help="默认与输入同名 .rrd")
    ap.add_argument("--stride", type=int, default=2, help="每 N 帧记一次")
    ap.add_argument("--axis-len", type=float, default=0.012,
                    help="关节朝向坐标轴长度（米）；0=不画朝向")
    ap.add_argument("--retarget", action="store_true",
                    help="离线复算 retarget 并渲染机器人手网格（不连设备）")
    ap.add_argument("--side", default="", help="left/right；--retarget 时用，留空按 frame_id 猜")
    ap.add_argument("--hand-model", default="wuji_hand",
                    choices=["wuji_hand", "wuji_hand_2"])
    ap.add_argument("--mjcf", default="")
    ap.add_argument("--tactile-max", type=float, default=0.0,
                    help="触觉色标上限；0=按本次录制的 p99 自动定标。"
                         "跨录制比较时给固定值（如 1.0）")
    ap.add_argument("--contact-thresh", type=float, default=0.05,
                    help="统计\"有接触 taxel 数\"的阈值")
    args = ap.parse_args()

    import rerun as rr
    import rerun.blueprint as rrb

    data = read_mcap(args.mcap)
    if not data:
        raise SystemExit("MCAP 里没有可识别的 topic：%s" % args.mcap)
    print("读到 %s" % ", ".join("%s×%d" % (t, len(v)) for t, v in data.items()))

    sk = data.get("hand_skeleton", [])
    if not sk:
        raise SystemExit("没有 hand_skeleton，画不了手")
    side = args.side or ("left" if sk[0][1]["header"].get("frame_id", "").startswith("l")
                         else "right")
    names = [j["name"] for j in sk[0][1]["joints"]]

    out = args.out or (os.path.splitext(args.mcap)[0] + ".rrd")
    rr.init("wuji_mcap_%s" % os.path.splitext(os.path.basename(args.mcap))[0])
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # ---- 可选：离线复算 retarget，渲染机器人手 ----
    geoms = m = d = sess = None
    if args.retarget:
        import mujoco
        from wuji_sdk import Handedness
        from retarget_compat import HandModel, RetargetSession

        mjcf = args.mjcf or resolve_mjcf(args.hand_model, side)
        m = mujoco.MjModel.from_xml_path(mjcf)
        d = mujoco.MjData(m)
        geoms = _mesh_geoms(m, mujoco)
        _log_hand_static(rr, "/world/robot", geoms, [200, 200, 210, 255])
        hm = (HandModel.WujiHand2 if args.hand_model == "wuji_hand_2"
              else HandModel.WujiHand)
        sess = RetargetSession.for_hand(
            hm, side=(Handedness.Right if side == "right" else Handedness.Left))
        sess.reset()
        print("  retarget: %s / %s，机器人手网格 %d 块"
              % (args.hand_model, os.path.basename(mjcf), len(geoms)))

    # 触觉/EMF 按最近时间戳对齐到骨架帧
    tac = data.get("tactile", [])
    emf = data.get("emf_poses", [])
    tac_t = [t for t, _ in tac]
    emf_t = [t for t, _ in emf]

    def nearest(times, arr, t):
        if not times:
            return None
        i = int(np.searchsorted(times, t))
        i = min(max(i, 0), len(arr) - 1)
        if i > 0 and abs(times[i - 1] - t) < abs(times[i] - t):
            i -= 1
        return arr[i][1]

    # 色标上限：实测单帧峰值常只有 0.4~0.5，固定用 1.0 会让颜色停在蓝青段、
    # 浪费一半动态范围，看着像"没接触"。默认按本次录制的 p99 定标。
    tac_max = args.tactile_max
    if tac_max <= 0 and tac:
        allv = np.concatenate([np.asarray(x[1]["data"], np.float32) for x in tac[::10]])
        allv = allv[allv >= 0]
        tac_max = float(np.percentile(allv, 99)) if allv.size else 1.0
        tac_max = max(tac_max, 0.05)
        print("  触觉色标上限（p99 自动）= %.3f" % tac_max)

    t0 = sk[0][0]
    for ts, s in sk[::args.stride]:
        rr.set_time("t", duration=(ts - t0) / 1e6)
        xyz, quat, conf = _pose_arrays(s["joints"])

        cols = np.array([[255, 140, 0] if c >= 0.3 else [255, 40, 40] for c in conf],
                        np.uint8)
        rr.log("/world/skeleton/joints", rr.Points3D(xyz, radii=0.005, colors=cols))
        rr.log("/world/skeleton/bones",
               rr.LineStrips3D([[xyz[a], xyz[b]] for a, b in SK_CONN],
                               radii=0.002, colors=[255, 170, 60]))
        # 关节朝向：把每个四元数展成三根轴线，否则 21 个点看不出旋转信息
        if args.axis_len > 0:
            segs, scols = [], []
            for p, q in zip(xyz, quat):
                R = _quat_to_mat(q)
                for k, c in enumerate(([255, 80, 80], [80, 255, 80], [80, 80, 255])):
                    segs.append([p, p + R[:, k] * args.axis_len])
                    scols.append(c)
            rr.log("/world/skeleton/axes", rr.LineStrips3D(segs, radii=0.0008,
                                                           colors=scols))
        for nm, c in zip(names, conf):
            rr.log("/conf/%s" % nm, rr.Scalars(float(c)))

        e = nearest(emf_t, emf, ts)
        if e:
            exyz, equat, econf = _pose_arrays(e["poses"])
            rr.log("/world/emf/coils", rr.Points3D(exyz, radii=0.006,
                                                   colors=[80, 200, 255]))
            segs, scols = [], []
            for p, q in zip(exyz, equat):
                R = _quat_to_mat(q)
                for k, c in enumerate(([255, 80, 80], [80, 255, 80], [80, 80, 255])):
                    segs.append([p, p + R[:, k] * args.axis_len * 1.6])
                    scols.append(c)
            rr.log("/world/emf/axes", rr.LineStrips3D(segs, radii=0.001, colors=scols))

        t = nearest(tac_t, tac, ts)
        if t and len(t.get("data") or []) == TACTILE_ROWS * TACTILE_COLS:
            raw = np.asarray(t["data"], np.float32)
            rr.log("/tactile/grid", rr.Image(tactile_rgb(raw, tac_max)))
            valid = raw[raw >= 0]
            # 有了标量才能在时间轴上找到"什么时候按下去了"，光看图得逐帧翻
            rr.log("/tactile_stat/peak", rr.Scalars(float(valid.max()) if valid.size else 0.0))
            rr.log("/tactile_stat/contact_taxels",
                   rr.Scalars(float((valid > args.contact_thresh).sum())))

        if sess is not None and xyz.shape == (21, 3):
            q20 = np.asarray(sess.step(xyz.astype(np.float32)), float)
            _log_hand_pose(rr, "/world/robot", geoms, m, d, q20, __import__("mujoco"),
                           m.nq)
            for i in range(min(20, len(q20))):
                rr.log("/retarget/j%02d" % i, rr.Scalars(float(q20[i])))

    rr.save(out, default_blueprint=_blueprint(rrb, bool(tac), bool(emf),
                                              sess is not None))
    print("→ %s（打开： ./venv312/bin/rerun %s）" % (out, out))
    return 0


def _quat_to_mat(q):
    """xyzw 四元数 → 3×3 旋转矩阵（列向量是局部 x/y/z 轴在父系下的方向）。

    先归一化：设备给的四元数是单位的，但 JSON 里是 f32 截断值，直接用会让
    R 偏离正交约 5e-5，画出来的朝向轴长度也会跟着漂。
    """
    q = np.asarray(q, np.float64)
    n = np.linalg.norm(q)
    if n > 1e-12:
        q = q / n
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float32)


if __name__ == "__main__":
    sys.exit(main())
