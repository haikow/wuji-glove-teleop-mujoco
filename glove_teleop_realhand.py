#!/usr/bin/env python3
"""带手套实时遥操 → **真机手**（一代 Wuji Hand / 二代 Wuji Hand 2，自动识别）。

链路 = Wuji Glove hand_skeleton (21,3)
      → RetargetSession.step()（SDK 内置 retargeting，输出 20 维固件序关节命令）
      → 真手（一代走 realtime_controller，二代走 MIT joint_command）。

与官方 `wuji-sdk/examples/python/retargeting/1.teleop_real.py` 同一套算法与驱动方式，
这里加了 argparse（选手套 / 手型 / 时长）和 `--keep-user`（用你标定的 per-user 手模型）。

⚠️ 安全：会**使能真手并让它跟随你的手实时运动**。跑之前把手固定好、周围无障碍、急停/断电在手边。
   Ctrl+C 或到时会自动 disable。

用法：
    python glove_teleop_realhand.py                       # 自动扫描；内置 URDF；跑到 Ctrl+C
    python glove_teleop_realhand.py --glove-sn <SN> --side right --seconds 20
    python glove_teleop_realhand.py --keep-user           # 用当前已标定用户的 right_hand.urdf

版本说明：2026.8.3 起 retargeting 原生化，`HandModel`/`RetargetSession` 顶层导入、
无需 `[retarget]` extra；≤7.15 走 `wuji_sdk.retargeting` 子模块。本脚本经 `retarget_compat` 兼容两者。
"""
import argparse
import contextlib
import time

import numpy as np

from wuji_sdk import (SdkManager, Handedness, JointCommand, LowPass,
                      WujiHand, WujiHand2, WujiGlove, ConnectOptions)
from retarget_compat import HandModel, RetargetSession


def read_keypoints(sub):
    """取 glove 最新一帧 hand_skeleton 的 (21,3) float32；无新帧返回 None。

    glove 发帧比消费快，recv() 返回最旧未读帧——每 tick 只取一帧会越落越后。
    这里排空队列只留最新，避免延迟累积。
    """
    latest = None
    while True:
        f = sub.recv()
        if f is None:
            break
        latest = f
    if latest is None:
        return None
    return np.array([j.pose.position for j in latest.joints], dtype=np.float32)


def teleop_loop(glove, hand, model, side, send, seconds, fps):
    sess = RetargetSession.for_hand(model, side=side)
    sub = glove.hand_skeleton().subscribe()
    budget = 1.0 / fps
    last = None
    t0 = time.monotonic()
    n = 0
    forever = seconds <= 0
    while forever or (time.monotonic() - t0 < seconds):
        fs = time.monotonic()
        kp = read_keypoints(sub)
        if kp is None:
            kp = last            # 无新帧：保持上一姿势
        if kp is None:
            time.sleep(budget)
            continue
        last = kp
        q = sess.step(kp)        # (20,) 固件序，直接下发
        send(q.tolist())
        n += 1
        dt = time.monotonic() - fs
        if dt < budget:
            time.sleep(budget - dt)
    print(f"teleop done: {n} frames")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove-sn", default="", help="手套 SN；留空自动挑扫到的第一只手套")
    ap.add_argument("--hand-sn", default="", help="真手 SN；多只手同时在总线上时用它指定目标（一代手 SN 形如 347A…）")
    ap.add_argument("--side", default="", help="left/right；留空自动读手套 hand_side()")
    ap.add_argument("--seconds", type=float, default=0.0, help="遥操时长秒；<=0 表示跑到 Ctrl+C")
    ap.add_argument("--fps", type=int, default=120, help="控制循环上限频率")
    ap.add_argument("--effort", type=float, default=1.5, help="电流上限 A")
    ap.add_argument("--keep-user", action="store_true",
                    help="保留当前 SDK 用户（加载其 per-user 标定手模型）；默认切到内置 URDF")
    args = ap.parse_args()

    mgr = SdkManager.instance()

    # 默认照官方：切默认用户跑内置 URDF（更稳）；--keep-user 则用当前已标定用户。
    previous_user = mgr.current_user()
    if args.keep_user:
        print(f"keep-user: 使用当前用户 {previous_user.get('display_name')}"
              f"（is_default={previous_user.get('is_default')}）的标定手模型")
    else:
        with contextlib.suppress(Exception):
            mgr.switch_to_default_user()
        print("使用内置默认 URDF（要用自己的标定加 --keep-user）")

    exit_code = 0
    try:
        no_bridge = ConnectOptions(enable_bridge=False)
        hand = glove = None
        # scan() 只给 SN/transport，不给型号——逐个连接后按语义类型分类。
        for d in mgr.scan():
            dev = mgr.connect(sn=d.sn, device_name=d.sn, options=no_bridge)
            if isinstance(dev, (WujiHand, WujiHand2)):
                if not args.hand_sn or dev.serial_number == args.hand_sn:
                    hand = dev            # 多只手时用 --hand-sn 指定，否则取扫到的最后一只
            elif isinstance(dev, WujiGlove):
                if not args.glove_sn or dev.serial_number == args.glove_sn:
                    glove = dev

        if hand is None or glove is None:
            print("未找到真手" if hand is None else "未找到手套")
            mgr.disconnect_all()
            return 1

        side = (args.side or str(glove.hand_side().get()).lower())
        hd = Handedness.Right if side.startswith("r") else Handedness.Left
        is_hand2 = isinstance(hand, WujiHand2)
        print(f"hand={type(hand).__name__}  glove={glove.serial_number}  side={side}")

        # 配置 + 使能
        if is_hand2:
            hand.effort_limit().set(args.effort)
            hand.mit_params().set((3.0, 0.05))   # (kp, kd) MIT，广播全关节
            hand.enable()
            model = HandModel.WujiHand2
        else:
            hand.set_all_effort_limit(args.effort)
            hand.enable()
            model = HandModel.WujiHand

        print("使能完成，开始遥操（Ctrl+C 停）...")
        try:
            if is_hand2:
                pub = hand.joint_command().publish()
                teleop_loop(glove, hand, model, hd,
                            lambda q: pub.send([JointCommand(p, 0.0, 0.0) for p in q]),
                            args.seconds, args.fps)
            else:
                with hand.realtime_controller(LowPass(cutoff_hz=5.0)) as ctrl:
                    teleop_loop(glove, hand, model, hd,
                                ctrl.set_target_position, args.seconds, args.fps)
        except KeyboardInterrupt:
            pass
        finally:
            with contextlib.suppress(Exception):
                hand.disable()
            mgr.disconnect_all()
    finally:
        # 恢复原用户（若切过默认）
        if not args.keep_user:
            with contextlib.suppress(Exception):
                mgr.switch_user(previous_user["user_id"])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
