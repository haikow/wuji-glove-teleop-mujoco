#!/usr/bin/env python3
"""Episode 录制：obs 走 SDK 的 MCAP 录制器，action 走旁路，按 header.seq 精确对齐。

容器分工（为什么是这个结构）：
  - **obs → `obs.mcap`**：直接用 `wuji_sdk.TopicRecorder`（LZ4 压缩、自描述 jsonschema、
    Foxglove 可开、自带 QualityMetrics 丢帧/同步率统计）。这是这套栈的 house format，
    没必要自己再造一个 JSONL。
  - **action → `action.jsonl`**：`TopicRecorder.record()` 只吃 `Subscription`（设备话题），
    而 retarget action 是 host 侧算出来的、没有对应设备资源路径（`publish()` 是往设备发、
    路径必须已存在），**存不进那个 MCAP**。所以单独落一份，用 `hand_skeleton` 的
    `header.seq` 做 join 键。
  - seq 是设备侧帧号：实测两个并行订阅看到的 seq 完全一致（361/361），且
    `hand_skeleton` 与 `hand_joint_angles` 共用同一 seq 空间、时间戳 diff=0µs，
    所以 join 是精确的，不靠时间戳近似。

读回时 `tools/episode_format.iter_frames()` 会把 MCAP + action 旁路合成统一的帧结构，
下游 QC / 导出 / 训练看到的东西和自包含 JSONL 版本完全一样。

用法：
    # 录 5 条 10 秒的 demo，每条录完提示打 success 标签
    ./venv312/bin/python record_episode.py --glove-sn <SN> --task pick_cube \
        --seconds 10 --episodes 5 --label

    # 用自己标定过的手模型录（否则回落内置 URDF）
    ./venv312/bin/python record_episode.py --glove-sn <SN> --task pick_cube --keep-user

    # 顺便出预览视频
    MUJOCO_GL=egl ./venv312/bin/python record_episode.py --glove-sn <SN> \
        --task pick_cube --video

录完直接质检：
    python tools/qc_episode.py data/episodes --route link
"""
import argparse
import asyncio
import os
import select
import sys
import time

import numpy as np
import mujoco

os.environ.setdefault("MUJOCO_GL", "egl")

from wuji_sdk import (  # noqa: E402
    ConnectOptions, Handedness, JointCommand, SdkManager, TopicRecorder, WujiGlove,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 2026.8.3 起 retargeting 原生化、顶层导出；≤7.15 在 wuji_sdk.retargeting 子模块。
# 走仓库的兼容层，两种版本都能跑。
from retarget_compat import HandModel, RetargetSession  # noqa: E402
from tools.episode_format import McapEpisodeWriter, obs_mcap_path  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))

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


def _sdk_version():
    try:
        from importlib.metadata import version
        return version("wuji-sdk")
    except Exception:
        return "unknown"


def _hdr(fr):
    """(seq, timestamp_us)，老固件/老 SDK 没有 header 时返回 (None, None)。"""
    h = getattr(fr, "header", None)
    if h is None:
        return None, None
    return getattr(h, "seq", None), getattr(h, "timestamp_us", None)


def _drain(sub):
    """把订阅队列里的积压全丢掉，只保留最后一帧（不累计延迟）。"""
    last = None
    while (x := sub.recv()) is not None:
        last = x
    return last


def _read_key():
    """非阻塞读一行 stdin：None=没输入，""=直接回车(停止)，"i"=标记人工介入。"""
    if not sys.stdin.isatty():
        return None
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if not r:
        return None
    return sys.stdin.readline().strip().lower()


def connect_glove(mgr, sn, tries=4):
    """连手套；手套是单客户端，被上个窗口占着时等心跳超时后重试。"""
    for attempt in range(1, tries + 1):
        try:
            return mgr.connect(sn=sn, device_name="glove",
                               options=ConnectOptions(enable_bridge=False))
        except Exception as e:
            if "already exists" in str(e).lower() and attempt < tries:
                print(f"[warn] 手套被占用(上个进程没退干净)，等待重试 {attempt}/{tries} ...")
                try:
                    mgr.disconnect_all()
                except Exception:
                    pass
                time.sleep(4)
                continue
            raise


async def record_one(ep, sub_sk, sess, jlo, jhi, nq, seconds,
                     recorder=None, mcap_path=None,
                     renderer=None, mj=None, video_path=None,
                     warmup_frames=60, warmup_timeout_s=5.0, hand_pub=None):
    """录一条 episode。返回 (stats, RecordingSummary|None)。

    recorder=None 时只写 action 旁路、不录 MCAP（离线测试用）。
    hand_pub 不为 None 时把 action 同步下发给真机手（joint_command）。
    """
    writer = None
    if renderer is not None and video_path:
        import imageio
        writer = imageio.get_writer(video_path, fps=30, macro_block_size=None)

    m, d, cam = (mj if mj else (None, None, None))
    sess.reset()

    # 预热（必须在 recorder.start() 之前，免得这段脏数据进 MCAP）：
    #   订阅后数据流有一段不稳定期；而且首次 sess.step() / 首次 render 是冷路径，实测
    #   吃掉 ~0.24s，这段时间设备又产了约 29 帧，于是头两帧的设备时间戳凭空拉开一个
    #   0.24s 的"假 gap"，会让每条 episode 都被 QC 判 gap+dropout。
    #   按"收到的帧数"而不是墙钟计：首帧本身可能就晚于任何固定时长的预热窗口。
    warmed = 0
    if warmup_frames > 0:
        tw = time.time()
        while warmed < warmup_frames and time.time() - tw < warmup_timeout_s:
            wf = _drain(sub_sk)
            if wf is not None:
                warmed += 1
                wkp = np.asarray([j.pose.position for j in wf.joints], dtype=np.float32)
                if wkp.shape == (21, 3):
                    wact = np.clip(sess.step(wkp), jlo[:nq], jhi[:nq])
                    if renderer is not None:
                        d.qpos[:nq] = wact[:nq]
                        mujoco.mj_forward(m, d)
                        renderer.update_scene(d, cam)
                        renderer.render()
            await asyncio.sleep(0.002)
        if warmed < warmup_frames:
            print(f"[warn] 预热只收到 {warmed}/{warmup_frames} 帧就超时，开头可能仍有 gap")

    handle = None
    if recorder is not None and mcap_path:
        handle = await recorder.start(mcap_path)
        # 丢掉 start() 之前就躺在订阅队列里的那一帧：它的 obs 没进 MCAP，
        # 为它写的 action 会变成 join 不上的孤儿（实测每条 episode 恰好多出 1 条）。
        _drain(sub_sk)

    last_seq = None
    no_seq_warned = False
    t0 = time.time()
    last_status = t0
    stats = {"actions": 0, "polls": 0, "skipped_dup": 0, "warmup_frames": warmed,
             "interventions": [], "send_errors": 0}
    stop_reason = "seconds"
    try:
        while True:
            now = time.time()
            if seconds and now - t0 >= seconds:
                stop_reason = "seconds"
                break
            key = _read_key()
            if key is not None:
                if key == "i":
                    # Human-Gated / DAgger 用：录制中标一段人工介入
                    stats["interventions"].append(round(now, 6))
                    print("\r  [intervention @ %.1fs]%s" % (now - t0, " " * 40))
                else:
                    stop_reason = "operator"
                    break

            stats["polls"] += 1
            fr = _drain(sub_sk)
            if fr is None:
                await asyncio.sleep(0.002)
                continue

            seq, t_dev_us = _hdr(fr)
            if seq is None:
                if not no_seq_warned:
                    print("[warn] 帧不带 header.seq，无法与 MCAP join，action 会全丢")
                    no_seq_warned = True
            elif seq == last_seq:
                stats["skipped_dup"] += 1
                await asyncio.sleep(0.002)
                continue
            last_seq = seq

            kp = np.asarray([j.pose.position for j in fr.joints], dtype=np.float32)
            if kp.shape != (21, 3):
                continue

            # retarget 已经在目标手型自己的 URDF 限位内解算，输出可直接下发；
            # 这里的 clip 只服务 MuJoCo 显示（本仓库 MJCF 只有一代手，用二代 profile 时
            # 限位对不上）。action 记录并下发的是**未经 MJCF clip 的原始解**，
            # action_raw_max_ovr 则量出它超出本地 MJCF 多少 —— 那正是手型不匹配的证据。
            raw = np.asarray(sess.step(kp), dtype=np.float64)
            act = raw
            disp = np.clip(raw, jlo[:len(raw)], jhi[:len(raw)])
            ovr = float(np.max(np.abs(raw - disp))) if raw.size else 0.0
            if hand_pub is not None:
                try:
                    hand_pub.send([JointCommand(float(v), 0.0, 0.0) for v in act[:20]])
                except Exception as e:
                    stats["send_errors"] += 1
                    if stats["send_errors"] == 1:
                        print("\r[warn] joint_command 下发失败: %s%s" % (e, " " * 20))
            ep.write_action(seq=seq, action=act, t_dev_us=t_dev_us, t_host=now,
                            action_raw_max_ovr=ovr)
            stats["actions"] += 1

            if renderer is not None:
                d.qpos[:nq] = disp[:nq]          # 显示用 clip 后的值
                mujoco.mj_forward(m, d)
                renderer.update_scene(d, cam)
                if writer is not None:
                    writer.append_data(renderer.render())

            if now - last_status >= 0.5:
                el = now - t0
                sys.stdout.write("\r  rec %5.1fs  actions=%5d  %5.1f Hz   "
                                 % (el, stats["actions"], stats["actions"] / max(el, 1e-3)))
                sys.stdout.flush()
                last_status = now
            await asyncio.sleep(0)
    except KeyboardInterrupt:
        stop_reason = "interrupt"
    finally:
        if writer is not None:
            writer.close()

    summary = None
    if handle is not None:
        summary = await handle.stop()
    stats["stop_reason"] = stop_reason
    stats["elapsed_s"] = time.time() - t0
    sys.stdout.write("\r" + " " * 78 + "\r")
    return stats, summary


async def main():
    ap = argparse.ArgumentParser(description="episode 化录制（obs.mcap + action 旁路）")
    ap.add_argument("--glove-sn", required=True,
                    help="手套 SN；不设默认值，避免真实设备号进公开仓库")
    ap.add_argument("--hand-sn", default="",
                    help="真机手 SN；给了就使能真机、下发 joint_command，并把 "
                         "joint_states 一起录进 obs.mcap（用于算跟踪误差）")
    ap.add_argument("--no-enable", action="store_true",
                    help="连真机手但不使能、不下发（只录 joint_states 反馈）")
    ap.add_argument("--task", required=True, help="任务名，进 meta.task，导出时按它分组")
    ap.add_argument("--hand-model", default="wuji_hand",
                    choices=["wuji_hand", "wuji_hand_2"])
    ap.add_argument("--side", default="", help="left/right；留空自动读手套")
    ap.add_argument("--out-dir", default="data/episodes")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="每条 episode 时长；0=一直录到回车/Ctrl-C")
    ap.add_argument("--episodes", type=int, default=1,
                    help="连录几条（复用同一个手套连接和 TopicRecorder 配置）")
    ap.add_argument("--user", default="",
                    help="切到这个 SDK 用户（user_id 或显示名）再连接，加载它的 per-user "
                         "标定手模型。标定见 tools/calibrate_glove.py")
    ap.add_argument("--keep-user", action="store_true",
                    help="沿用当前 SDK 用户，不做切换；默认切 default 用内置 URDF")
    ap.add_argument("--no-joint-angles", action="store_true")
    ap.add_argument("--tactile", action="store_true", help="MCAP 里也录触觉")
    ap.add_argument("--compression", default="lz4", choices=["lz4", "zstd", "none"])
    ap.add_argument("--video", action="store_true",
                    help="每条 episode 顺带渲染 preview.mp4（EGL 离屏）")
    ap.add_argument("--label", action="store_true",
                    help="每条录完提示打 success 标签 (y/n/回车跳过)")
    ap.add_argument("--warmup-frames", type=int, default=60,
                    help="开录前先丢弃多少帧，等数据流和计算路径都热起来（默认 60≈0.5s@120Hz）")
    ap.add_argument("--notes", default="")
    ap.add_argument("--mjcf", default="", help="手动指定 MJCF 路径（默认按 --hand-model 选）")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    args = ap.parse_args()

    mgr = SdkManager.instance()
    previous_user = None
    try:
        previous_user = mgr.current_user()
    except Exception as e:
        print(f"[warn] current_user 查询失败: {e}")
    if args.user:
        target = None
        for u in mgr.list_users():
            if args.user in (u["user_id"], u["display_name"]):
                target = u
                break
        if target is None:
            raise SystemExit("找不到 SDK 用户：%s（可用：%s）"
                             % (args.user, ", ".join(u["display_name"]
                                                     for u in mgr.list_users())))
        if target.get("is_default"):
            raise SystemExit("--user 指到了默认用户，它会回落内置 URDF、忽略标定")
        # 必须在 connect 之前切：设备连上后才会按当前用户加载参数与手模型
        mgr.switch_user(target["user_id"])
        print(f"已切到 SDK 用户 {target['display_name']} ({target['user_id']})")
    elif args.keep_user:
        print(f"keep-user: 沿用当前用户 {previous_user and previous_user.get('display_name')}")
        previous_user = None          # 没切用户，收尾不用还原
    else:
        try:
            mgr.switch_to_default_user()
            print("已切到 SDK 默认用户（内置 URDF，忽略标定）；"
                  "要用标定过的手型请加 --user <用户名>")
        except Exception as e:
            print(f"[warn] switch_to_default_user 失败: {e}")

    g = connect_glove(mgr, args.glove_sn)
    side = args.side or str(g.hand_side().get()).lower()
    hd = Handedness.Right if side == "right" else Handedness.Left
    print(f"glove {args.glove_sn} connected, side={side}")

    time_sync = None
    try:
        r = g.sync_time()
        time_sync = {"offset_us": int(r.offset_us), "round_trip_us": int(r.round_trip_us)}
        print(f"time sync: offset={time_sync['offset_us']}us rtt={time_sync['round_trip_us']}us")
    except Exception as e:
        print(f"[warn] sync_time 失败（将只有主机时钟）: {e}")

    # 标定有没有生效的硬证据：offline_pipeline 会按当前 SDK 用户解析出实际用的 URDF 来源。
    #   builtin_default   = 内置模型（默认用户永远是这个，标定被忽略）
    #   calibration_file  = 该用户 users/<uid>/models/ 下的标定模型
    #   override          = 有人显式设了 hand_model_path 覆盖
    # 注意 hand_model_path() 是"覆盖槽"而不是"当前模型"，没设 override 时会直接报
    # Path not found，不能拿它当判据。
    urdf_source = urdf_source_path = None
    try:
        pipe = WujiGlove.offline_pipeline(args.glove_sn, side)
        urdf_source = str(pipe.urdf_source)
        urdf_source_path = pipe.urdf_source_path
        print(f"hand model source: {urdf_source}  {urdf_source_path or ''}")
        if urdf_source == "builtin_default":
            print("[warn] 用的是内置 URDF、没吃标定 —— retarget 会对不准，"
                  "请用 --user <标定过的用户>")
    except Exception as e:
        print(f"[warn] urdf_source 读取失败: {e}")

    cur_user = {}
    try:
        cur_user = mgr.current_user() or {}
    except Exception:
        pass

    # ---- 真机手（可选）----
    hand = hand_pub = None
    hand_info = None
    if args.hand_sn:
        hand = mgr.connect(sn=args.hand_sn, device_name="hand",
                           options=ConnectOptions(enable_bridge=False))
        n_online = None
        try:
            n_online = int(hand.online_joints_count().get())
        except Exception as e:
            print(f"[warn] online_joints_count 读取失败: {e}")
        hside = None
        try:
            hside = str(hand.handedness().get()).lower()
        except Exception:
            pass
        print(f"hand {args.hand_sn} connected, side={hside}, online_joints={n_online}")
        if hside and side and hside != side:
            raise SystemExit(f"手套是 {side} 手、真机手是 {hside} 手，手性不匹配，拒绝下发")
        if n_online is not None and n_online < 20:
            print(f"[warn] 只有 {n_online}/20 关节在线，缺失关节不会跟随")
        hand_info = {"hand_sn": args.hand_sn, "online_joints": n_online,
                     "handedness": hside, "enabled": not args.no_enable}
        if not args.no_enable:
            hand.enable()
            print("hand enabled（退出时会自动 disable）")
            hand_pub = hand.joint_command().publish()

    # 两套订阅：一套交给 MCAP 录制器，一套自己读来算 action。
    # 实测同一资源开两个订阅互不干扰，seq 完全一致。
    recorder = TopicRecorder(compression=args.compression)
    channels = ["hand_skeleton"]
    recorder.record(g.hand_skeleton().subscribe())
    if not args.no_joint_angles:
        recorder.record(g.hand_joint_angles().subscribe())
        channels.append("hand_joint_angles")
    if args.tactile:
        recorder.record(g.tactile().subscribe())
        channels.append("tactile")
    if hand is not None:
        # 真机反馈也是设备话题，可以直接进同一个 MCAP（~999Hz，读回时按时间戳并到帧上）
        recorder.record(hand.joint_states().subscribe())
        channels.append("joint_states")
    sub_sk = g.hand_skeleton().subscribe()

    hm = (HandModel.WujiHand2 if args.hand_model == "wuji_hand_2"
          else HandModel.WujiHand)
    sess = RetargetSession.for_hand(hm, side=hd)

    mjcf = resolve_mjcf(args.hand_model, side, args.mjcf)
    print(f"MJCF: {mjcf}")
    m = mujoco.MjModel.from_xml_path(mjcf)
    d = mujoco.MjData(m)
    jlo, jhi = m.jnt_range[:, 0].copy(), m.jnt_range[:, 1].copy()
    joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]

    renderer = mj = None
    if args.video:
        renderer = mujoco.Renderer(m, args.h, args.w)
        cam = setup_camera(m, mujoco)
        mj = (m, d, cam)

    env = {
        "glove_sn": args.glove_sn,
        "hand_model": args.hand_model,
        "urdf_source": urdf_source,
        "urdf_source_path": urdf_source_path,
        "user_id": cur_user.get("user_id"),
        "user_display_name": cur_user.get("display_name"),
        # 默认用户 = 内置 URDF、忽略标定；只有 --keep-user 才吃 per-user 标定模型
        # 如实判定：当前 SDK 用户不是默认用户才可能加载 per-user 标定模型
        "calibrated": not bool(cur_user.get("is_default", not (args.user or args.keep_user))),
        "user_mode": ("user:%s" % args.user) if args.user
                     else ("keep_user" if args.keep_user else "default_user"),
        "sdk_version": _sdk_version(),
        "time_sync": time_sync,
        "action_space": {"name": "retarget_qpos", "dim": int(m.nq),
                         "unit": "rad", "joint_names": joint_names},
        "obs_space": {"container": "mcap", "channels": channels,
                      "skeleton": [21, 3], "join_key": "hand_skeleton.header.seq"},
        "mcap_compression": args.compression,
        "mjcf": mjcf,
        "hand": hand_info,
        "recorder": "record_episode.py",
    }

    os.makedirs(args.out_dir, exist_ok=True)
    done = []
    try:
        for k in range(args.episodes):
            if sys.stdin.isatty():
                prompt = (f"\n[{k+1}/{args.episodes}] 回车开始录制"
                          + (f"（{args.seconds:.0f}s 自动停，录制中回车可提前停）"
                             if args.seconds else "（录制中回车停止）") + " > ")
                try:
                    input(prompt)
                except EOFError:
                    break
            ep = McapEpisodeWriter(args.out_dir, side=side, task=args.task, env=env)
            vp = os.path.join(ep.path, "preview.mp4") if args.video else None
            st, summary = await record_one(
                ep, sub_sk, sess, jlo, jhi, m.nq, args.seconds,
                recorder=recorder, mcap_path=obs_mcap_path(ep.path),
                renderer=renderer, mj=mj, video_path=vp,
                warmup_frames=args.warmup_frames, hand_pub=hand_pub)

            success = None
            if args.label and sys.stdin.isatty():
                try:
                    a = input("  success? [y/n/回车=未标] > ").strip().lower()
                except EOFError:
                    a = ""
                success = True if a.startswith("y") else (False if a.startswith("n") else None)
            ep.env["stop_reason"] = st["stop_reason"]
            ep.env["warmup_frames_discarded"] = st["warmup_frames"]
            ep.env["intervention_times"] = st["interventions"]
            if hand_pub is not None:
                ep.env["command_send_errors"] = st["send_errors"]
            if summary is not None:
                q = summary.quality
                # 注意 QualitySummary 的 repr 显示的是 dropped/drop_rate，
                # 实际属性名却是 dropped_frames/frame_drop_rate，别照 repr 写。
                ep.env["sdk_quality"] = {
                    "total_frames": summary.total_frames,
                    "file_size": summary.file_size,
                    "duration_s": round(summary.duration_s, 3),
                    "dropped_frames": q.dropped_frames,
                    "frame_drop_rate": q.frame_drop_rate,
                    "sync_rate": q.sync_rate,
                    "avg_sync_offset_ms": q.avg_sync_offset_ms,
                    "max_sync_offset_ms": q.max_sync_offset_ms,
                    "spc_alert_count": q.spc_alert_count,
                }
            ep.finalize(success=success, intervention=bool(st["interventions"]),
                        notes=args.notes)
            done.append(ep.path)
            sdk_n = summary.total_frames if summary is not None else 0
            print("  → %s  action=%d  mcap=%d frames  %.1f Hz  (%s)"
                  % (os.path.basename(ep.path), st["actions"], sdk_n,
                     st["actions"] / max(st["elapsed_s"], 1e-3), st["stop_reason"]))
            if st["stop_reason"] == "interrupt":
                break
    except KeyboardInterrupt:
        print("\n中断，已保存完成的 episode")
    finally:
        if hand_pub is not None:
            try:
                hand_pub.close()
            except Exception:
                pass
        if hand is not None and not args.no_enable:
            try:
                hand.disable()
                print("hand disabled")
            except Exception as e:
                print(f"[warn] hand.disable 失败，请手动断电确认: {e}")
        if renderer is not None:
            renderer.close()
        if previous_user is not None:
            try:
                mgr.switch_user(previous_user["user_id"])
            except Exception as e:
                print(f"[warn] 还原用户失败: {e}")
        try:
            mgr.disconnect_all()
        except Exception:
            pass

    print(f"\n共 {len(done)} 条 episode → {args.out_dir}")
    print(f"下一步质检： python tools/qc_episode.py {args.out_dir} --route link")
    # os._exit 跳过解释器清理（SDK 后台线程退不干净时会挂住），但也跳过 stdout flush，
    # 不手动 flush 会吞掉最后几行输出。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
