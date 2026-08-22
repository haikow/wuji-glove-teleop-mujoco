#!/usr/bin/env python3
"""手套 IK 标定：建具名用户 → 切过去 → 跑引导式标定 → 校验产物落到该用户名下。

**为什么必须先建具名用户**：默认用户走的是 legacy 路径，加载时会强制回落内置 URDF、
忽略标定结果。只有具名用户的 `users/<uid>/models/` 才会被实时 IK 真正加载。

**hand_profile 要和你的真机手对齐**：`wujihand`=一代手模型，`wujihand2`=二代手模型
（SN 以 WH2 开头、SDK 类型为 WujiHand2 的是二代）。不传则两套都生成，实时 IK 默认仍用
`wujihand`，遥操时记得 `--hand-model wuji_hand_2`。

标定是**交互式**的：要按提示摆 4 个姿势并保持不动。请在终端里直接跑，才能看到实时提示：

    ./venv312/bin/python tools/calibrate_glove.py --glove-sn <SN> \\
        --user-name jues_20260822 --hand-profile wujihand2

跑完用它遥操：

    ./venv312/bin/python record_episode.py --glove-sn <SN> --hand-sn <手SN> \\
        --user jues_20260822 --hand-model wuji_hand_2 --task pick_cube
"""
import argparse
import os
import sys
import time

from wuji_sdk import ConnectOptions, SdkManager, WujiGlove, WujiHandProfile

# SDK 报出来的 step_name → 人话提示（名字对不上时原样显示）。
# 实测 caliber 要 6 个姿势：4 个捏合 + 四指弯 90° + 摊平张开。
STEP_HINTS = {
    "pinch_index": "拇指与食指指尖捏合",
    "pinch_middle": "拇指与中指指尖捏合",
    "pinch_ring": "拇指与无名指指尖捏合",
    "pinch_pinky": "拇指与小指指尖捏合",
    "four_finger_bend_90": "四指弯曲约 90°（拇指自然放）",
    "flat_open": "手掌摊平、五指伸直张开",
}


def _find_user(mgr, name_or_id):
    for u in mgr.list_users():
        if name_or_id in (u["user_id"], u["display_name"]):
            return u
    return None


def main():
    ap = argparse.ArgumentParser(description="手套 IK 标定（写进具名 SDK 用户）")
    ap.add_argument("--glove-sn", required=True)
    ap.add_argument("--user-name", required=True,
                    help="SDK 用户显示名；不存在就新建（标定产物落到这个用户名下）")
    ap.add_argument("--hand-profile", default="wujihand2",
                    choices=["wujihand", "wujihand2", "both"],
                    help="目标手型；both=两套都生成。二代手(WH2/WujiHand2)选 wujihand2")
    ap.add_argument("--skip-constraints", action="store_true",
                    help="跳过约束校验（标不过时应急用，产物质量会差）")
    ap.add_argument("--timeout-s", type=float, default=900.0)
    args = ap.parse_args()

    mgr = SdkManager.instance()

    # 1) 建 / 找具名用户，并在**连接之前**切过去
    u = _find_user(mgr, args.user_name)
    if u is None:
        u = mgr.create_user(display_name=args.user_name,
                            description="IK calibration %s" % time.strftime("%Y-%m-%d"))
        print("已新建 SDK 用户：%s (%s)" % (u["display_name"], u["user_id"]))
    else:
        print("复用已有 SDK 用户：%s (%s)" % (u["display_name"], u["user_id"]))
    if u.get("is_default"):
        raise SystemExit("不能用默认用户标定 —— 默认用户加载时会回落内置 URDF、忽略标定")
    mgr.switch_user(u["user_id"])
    print("已切换当前用户 → %s" % mgr.current_user()["display_name"])

    g = mgr.connect(sn=args.glove_sn, device_name="calib",
                    options=ConnectOptions(enable_bridge=False))
    side = str(g.hand_side().get()).lower()
    print("glove %s connected, side=%s" % (args.glove_sn, side))
    print("\n开始标定，按提示摆姿势并保持不动。Ctrl+C 中止。\n")

    state = {"step": None}

    def on_feedback(fb):
        step = fb.get("step_index")
        name = fb.get("step_name") or ""
        if step != state["step"]:
            state["step"] = step
            # step_index 是 0-based，显示成人看的 1-based
            print("\n[%s/%s] %s —— %s"
                  % ((step + 1) if isinstance(step, int) else step,
                     fb.get("step_total"), name,
                     STEP_HINTS.get(name, "按 Studio 里同名步骤的姿势做")))
        bits = [fb.get("state") or ""]
        if fb.get("hold_target"):
            bits.append("保持 %.1f/%.1fs" % (fb.get("hold_elapsed") or 0,
                                             fb["hold_target"]))
        if fb.get("collect_target"):
            bits.append("采集 %.1f/%.1fs" % (fb.get("collect_elapsed") or 0,
                                             fb["collect_target"]))
        if fb.get("frames_collected"):
            bits.append("%d 帧" % fb["frames_collected"])
        if fb.get("variance") is not None:
            bits.append("抖动 %.4f%s" % (fb["variance"],
                                         "" if fb.get("variance_ok", True) else " ✗手别动"))
        sys.stdout.write("\r  " + "  ".join(b for b in bits if b) + " " * 12)
        sys.stdout.flush()
        for h in (fb.get("hints") or []):
            print("\n  提示：%s" % h)

    profile = None if args.hand_profile == "both" else WujiHandProfile(args.hand_profile)
    try:
        res = g.calibrate_blocking(skip_constraints=args.skip_constraints,
                                   timeout_s=args.timeout_s,
                                   hand_profile=profile,
                                   on_feedback=on_feedback)
    except KeyboardInterrupt:
        print("\n已中止，未写入标定结果")
        mgr.disconnect_all()
        return 130
    print("\n")

    # 2) 校验产物确实落到了这个用户名下
    print("标定完成：")
    print("  device_sn            %s" % res.get("device_sn"))
    print("  hand_profile         %s" % res.get("hand_profile"))
    print("  active_hand_profile  %s" % res.get("active_hand_profile"))
    print("  generated_profiles   %s" % res.get("generated_hand_profiles"))
    print("  poses_collected      %s" % res.get("poses_collected"))
    for k, v in (res.get("frames_per_pose") or {}).items():
        print("      %-28s %s 帧" % (k, v))
    su = res.get("sdk_user") or {}
    print("  sdk_user             %s (%s)" % (su.get("display_name"), su.get("user_id")))

    urdfs = res.get("calibrated_urdfs") or {}
    ok = False
    for prof, path in urdfs.items():
        under_user = ("users/%s" % u["user_id"]) in (path or "")
        ok = ok or under_user
        print("  urdf[%s] %s%s" % (prof, path, "" if under_user else "   ← 不在该用户目录下！"))
    if not urdfs:
        print("  [warn] 没有返回 calibrated_urdfs")
    if not ok:
        print("\n[warn] 标定产物似乎没落在 users/%s/models/ 下 —— "
              "实时 IK 可能仍会回落内置 URDF，遥操前请确认 hand_model_path" % u["user_id"])

    # 3) 校验标定是否真的会被加载。判据是 offline_pipeline 解析出的 urdf_source，
    #    **不是** hand_model_path()（那是"覆盖槽"，没设 override 时直接报
    #    Path not found，拿它当判据会误判）。
    mgr.disconnect_all()
    try:
        pipe = WujiGlove.offline_pipeline(args.glove_sn, side)
        print("\n实际生效的手模型来源：%s\n  %s"
              % (pipe.urdf_source, pipe.urdf_source_path or "(无)"))
        if pipe.urdf_source != "calibration_file":
            print("[warn] urdf_source 不是 calibration_file —— 标定没被采用")
        elif ("users/%s" % u["user_id"]) not in (pipe.urdf_source_path or ""):
            print("[warn] 生效的标定文件不在本用户目录下")
    except Exception as e:
        print("[warn] urdf_source 校验失败：%s" % e)
    hm = "wuji_hand_2" if (res.get("active_hand_profile") == "wujihand2"
                           or args.hand_profile == "wujihand2") else "wuji_hand"
    print("\n下一步遥操（注意 --user 和 --hand-model 要和标定时一致）：")
    print("  ./venv312/bin/python record_episode.py --glove-sn %s \\\n"
          "      --user %s --hand-model %s --task <任务名>"
          % (args.glove_sn, args.user_name, hm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
