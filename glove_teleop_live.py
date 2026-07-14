#!/usr/bin/env python3
"""带手套实时遥操 → MuJoCo（EGL 离屏渲染 + cv2 窗口，绕开 :1 上坏掉的 GLX）。

链路 = 客户方案 B 的计算侧：Wuji Glove hand_skeleton (21,3)
      → wuji_sdk.retargeting.RetargetSession.step() (闭源好算法)
      → MuJoCo 手模型，实时显示。

用法：
    MUJOCO_GL=egl python glove_teleop_live.py                 # 实时窗口，按 q/ESC 退出
    MUJOCO_GL=egl python glove_teleop_live.py --record out.mp4 --seconds 20
    MUJOCO_GL=egl python glove_teleop_live.py --headless --record out.mp4 --seconds 5   # 自测，不开窗
"""
import argparse
import os
import time

import numpy as np
import mujoco
import mujoco.viewer  # noqa: F401  官方交互窗口（--viewer）

os.environ.setdefault("MUJOCO_GL", "egl")

from wuji_sdk import SdkManager, Handedness, retargeting

MJCF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "wuji_hand_description", "mjcf")

# MediaPipe 手骨架连线（同 tuning_tool 的 SKELETON_CONNECTIONS）
SK_CONN = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),          # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),     # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),   # 无名
    (0, 17), (17, 18), (18, 19), (19, 20),   # 小指
    (5, 9), (9, 13), (13, 17),               # 掌横连
]
TIP_INDICES = {4, 8, 12, 16, 20}             # 指尖（大球）


# ---- MediaPipe 坐标变换（同 wuji_retargeting/mediapipe.py，用于把输入对齐到机器人腕系）----
OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float)


def _estimate_frame_from_hand_points(kp):
    pts = kp[[0, 5, 9], :]
    x_vec = pts[0] - pts[2]
    pts = pts - pts.mean(0, keepdims=True)
    _, _, v = np.linalg.svd(pts)
    normal = v[2, :]
    x = x_vec - np.sum(x_vec * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)
    if np.sum(z * (pts[1] - pts[2])) < 0:
        normal, z = -normal, -z
    return np.stack([x, normal, z], axis=1)


def apply_mediapipe_transformations(kp, hand_type="right"):
    kp = kp - kp[0:1, :]
    rot = _estimate_frame_from_hand_points(kp)
    o2m = OPERATOR2MANO_RIGHT if hand_type.lower() == "right" else OPERATOR2MANO_LEFT
    return kp @ rot @ o2m


# MediaPipe index -> 机器人 body（robot_fk 白层用）；腕体=palm_link
ROBOT_FK_BODY = ["palm_link"] + [f"finger{f}_link{j}"
                                 for f in range(1, 6) for j in range(1, 5)]

# 层样式，逐值照搬 skeleton_drawer.DEFAULT_LAYER_CONFIG
LAYER_INPUT = dict(pt=np.array([1.0, 0.5, 0.0, 0.6], np.float32),
                   ln=np.array([1.0, 0.6, 0.2, 0.5], np.float32),
                   ps=0.004, lw=0.002)
LAYER_FK = dict(pt=np.array([1.0, 1.0, 1.0, 0.95], np.float32),
                ln=np.array([1.0, 1.0, 1.0, 0.8], np.float32),
                ps=0.005, lw=0.003)


def _draw_layer(scn, pts, cfg):
    """把一层 21 点骨架追加进场景（不清零 ngeom；点=球，连线=胶囊）。"""
    eye = np.eye(3).flatten()
    for p in pts:
        if scn.ngeom >= scn.maxgeom:
            return
        if not np.all(np.isfinite(p)):
            continue
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([cfg["ps"], 0, 0], np.float64),
                            np.ascontiguousarray(p, np.float64), eye, cfg["pt"])
        scn.ngeom += 1
    for a, b in SK_CONN:
        if scn.ngeom >= scn.maxgeom:
            return
        if not (np.all(np.isfinite(pts[a])) and np.all(np.isfinite(pts[b]))):
            continue
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3), np.zeros(3), eye, cfg["ln"])
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, cfg["lw"],
                             np.ascontiguousarray(pts[a], np.float64),
                             np.ascontiguousarray(pts[b], np.float64))
        scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove-sn", default="WG1KA00260124117")
    ap.add_argument("--hand-model", default="wuji_hand",
                    choices=["wuji_hand", "wuji_hand_2"])
    ap.add_argument("--side", default="", help="left/right；留空自动读手套")
    ap.add_argument("--record", default="")
    ap.add_argument("--seconds", type=float, default=0.0, help="0=一直跑到按 q")
    ap.add_argument("--viewer", action="store_true",
                    help="用官方 mujoco.viewer 交互窗口（GLFW，可轨道/接触可视化）；默认走 cv2+EGL")
    ap.add_argument("--clean", action="store_true",
                    help="--viewer 时隐藏左右面板并关掉关节叠加层（只看手）；默认显示完整调试页")
    ap.add_argument("--show-input", action="store_true",
                    help="--viewer 时把手套输入的21点人手骨架（骨长+指尖球）叠加进场景（橙色，同 tuning_tool 输入层）")
    ap.add_argument("--input-offset", default="0,0.12,0.05",
                    help="输入骨架相对机器人手的摆放偏移 x,y,z（米）")
    ap.add_argument("--headless", action="store_true", help="不开窗（自测/录制用）")
    ap.add_argument("--keep-user", action="store_true",
                    help="不切 default 用户，保留当前已标定用户的 per-user 手模型"
                         "（做完 Studio/calibrate 标定后想看到自己标定效果时用这个）")
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=720)
    args = ap.parse_args()

    # 连手套
    from wuji_sdk import ConnectOptions
    mgr = SdkManager.instance()
    # 默认：切到 SDK 默认用户，跑"内置默认手 URDF"（通用、跟随稳，但忽略 per-user 标定）。
    # --keep-user：保留当前用户，加载其 per-user 标定手模型（标定后想看自己的零位/手型效果）。
    previous_user = None
    if args.keep_user:
        try:
            cur = mgr.current_user()
            print(f"keep-user: 使用当前已标定用户 {cur.get('display_name')} "
                  f"({cur.get('user_id')}) 的 per-user 手模型")
        except Exception as e:
            print(f"[warn] current_user query failed: {e}")
    else:
        try:
            previous_user = mgr.current_user()
            mgr.switch_to_default_user()
            print("switched to default SDK user (built-in URDF)；标定后想看自己效果请加 --keep-user")
        except Exception as e:
            print(f"[warn] switch_to_default_user failed: {e}")
    g = None
    for attempt in range(1, 5):
        try:
            g = mgr.connect(sn=args.glove_sn, device_name="glove",
                            options=ConnectOptions(enable_bridge=False))
            break
        except Exception as e:
            if "already exists" in str(e).lower() and attempt < 4:
                print(f"[warn] 手套被占用(可能上个窗口没关)，等心跳超时后重试 {attempt}/4 ...")
                try:
                    mgr.disconnect_all()
                except Exception:
                    pass
                time.sleep(4)
                continue
            raise
    side = args.side or str(g.hand_side().get()).lower()
    hd = Handedness.Right if side == "right" else Handedness.Left
    print(f"glove {args.glove_sn} connected, side={side}")

    sub = g.hand_skeleton().subscribe()
    model = (retargeting.HandModel.WujiHand2 if args.hand_model == "wuji_hand_2"
             else retargeting.HandModel.WujiHand)
    sess = retargeting.RetargetSession.for_hand(model, side=hd)
    sess.reset()

    m = mujoco.MjModel.from_xml_path(os.path.join(MJCF_DIR, f"{side}.xml"))
    d = mujoco.MjData(m)
    # 运动学显示：直接置关节角 + mj_forward，让手精确呈现 retarget 结果。
    # （MJCF 的 position 执行器 kp 很软 kp=2/1/0.8，用 ctrl+mj_step 会严重欠到位，
    #  握拳只curl一点点，看起来像"跟不上"——那是仿真欠驱动，不是 retarget 的问题。）
    jlo, jhi = m.jnt_range[:, 0].copy(), m.jnt_range[:, 1].copy()

    # ---- 官方 mujoco.viewer 交互窗口（GLFW）----
    if args.viewer:
        os.environ.pop("MUJOCO_GL", None)   # 交互窗口走 GLFW，不用 EGL 离屏
        last_kp = None
        # 白层(robot FK)与腕体对齐用的 body id
        fk_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ROBOT_FK_BODY]
        palm_id = fk_ids[0]
        can_align = palm_id >= 0
        # 默认显示左右面板；--clean 则隐藏面板+关掉叠加层（只看手）
        show_ui = not args.clean
        with mujoco.viewer.launch_passive(m, d, show_left_ui=show_ui, show_right_ui=show_ui) as v:
            v.cam.azimuth, v.cam.elevation, v.cam.distance = 180, -20, 0.42
            v.cam.lookat[:] = [0, 0, 0.05]
            # 关节可视化叠加层（青色轴 + 橙色标记）：默认开；--clean 关
            joint_on = not args.clean
            for _f in ("mjVIS_JOINT", "mjVIS_ACTUATOR"):
                fl = getattr(mujoco.mjtVisFlag, _f, None)
                if fl is not None:
                    v.opt.flags[fl] = joint_on
            v.sync()
            while v.is_running():
                fr = None
                while (x := sub.recv()) is not None:
                    fr = x
                if fr is not None:
                    kp = np.asarray([j.pose.position for j in fr.joints], dtype=np.float32)
                    if kp.shape == (21, 3):
                        last_kp = kp
                if last_kp is not None:
                    d.qpos[:m.nq] = np.clip(sess.step(last_kp), jlo, jhi)
                    mujoco.mj_forward(m, d)
                    if args.show_input and can_align:
                        scn = v.user_scn
                        scn.ngeom = 0
                        # 腕体(palm_link)世界位姿：把输入/目标 21 点从腕系搬到世界（同 tuning_tool）
                        wpos = d.xpos[palm_id]
                        wrot = d.xmat[palm_id].reshape(3, 3)
                        # 橙层：输入(经 mediapipe 变换到腕系) → 世界
                        tkp = apply_mediapipe_transformations(last_kp.astype(np.float64), side)
                        _draw_layer(scn, tkp @ wrot.T + wpos, LAYER_INPUT)
                        # 白层：机器人 FK（各 link 世界坐标，缺失置 nan 跳过）
                        fk = np.array([d.xpos[i] if i >= 0 else [np.nan] * 3
                                       for i in fk_ids], float)
                        _draw_layer(scn, fk, LAYER_FK)
                v.sync()
                time.sleep(0.008)
        if previous_user is not None:
            try:
                mgr.switch_user(previous_user["user_id"])
            except Exception:
                pass
        os._exit(0)

    ren = mujoco.Renderer(m, args.h, args.w)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = 180, -20, 0.42
    cam.lookat[:] = [0, 0, 0.05]

    # --show-input 也走 EGL 离屏路径（GLFW 在坏 GLX 的 :1 上崩，这条能出图/录像）
    fk_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ROBOT_FK_BODY]
    palm_id = fk_ids[0]
    can_align = args.show_input and palm_id >= 0

    cv2 = None
    if not args.headless:
        import cv2 as _cv2
        cv2 = _cv2
        cv2.namedWindow("Wuji Glove Teleop (q=quit)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Wuji Glove Teleop (q=quit)", args.w, args.h)

    writer = None
    if args.record:
        import imageio
        writer = imageio.get_writer(args.record, fps=30, macro_block_size=None)

    last_kp = None
    t0 = time.time()
    frames = 0
    last_fps_t = t0
    fps = 0.0
    try:
        while True:
            fr = None
            while (x := sub.recv()) is not None:
                fr = x
            if fr is not None:
                kp = np.asarray([j.pose.position for j in fr.joints], dtype=np.float32)
                if kp.shape == (21, 3):
                    last_kp = kp
            if last_kp is None:
                time.sleep(0.005)
                continue

            qpos = sess.step(last_kp)
            d.qpos[:m.nq] = np.clip(qpos, jlo, jhi)   # 运动学置位：精确呈现
            mujoco.mj_forward(m, d)
            ren.update_scene(d, cam)
            if can_align:
                scn = ren.scene
                wpos = d.xpos[palm_id]
                wrot = d.xmat[palm_id].reshape(3, 3)
                tkp = apply_mediapipe_transformations(last_kp.astype(np.float64), side)
                _draw_layer(scn, tkp @ wrot.T + wpos, LAYER_INPUT)   # 橙：人手输入
                fk = np.array([d.xpos[i] if i >= 0 else [np.nan] * 3
                               for i in fk_ids], float)
                _draw_layer(scn, fk, LAYER_FK)                       # 白：机器人 FK
            rgb = ren.render()

            frames += 1
            now = time.time()
            if now - last_fps_t >= 0.5:
                fps = frames / (now - t0)
                last_fps_t = now

            if writer is not None:
                writer.append_data(rgb)

            if cv2 is not None:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr, f"{side} hand  {fps:4.1f} fps  (q=quit)",
                            (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Wuji Glove Teleop (q=quit)", bgr)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            if args.seconds and now - t0 >= args.seconds:
                break
    finally:
        if writer is not None:
            writer.close()
        ren.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
        # 还原之前的 SDK 用户
        if previous_user is not None:
            try:
                mgr.switch_user(previous_user["user_id"])
            except Exception as e:
                print(f"[warn] restore user failed: {e}")
        print(f"done: {frames} frames, avg {frames/max(time.time()-t0,1e-3):.1f} fps"
              + (f", saved {args.record}" if args.record else ""))
        os._exit(0)


if __name__ == "__main__":
    main()
