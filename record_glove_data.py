#!/usr/bin/env python3
"""Wuji Glove 数据采集 → CSV + JSONL 落盘。

同步录制：hand_skeleton(21×3) + hand_joint_angles(21) + tactile，带主机接收时间戳。
用法：
    ./venv312/bin/python record_glove_data.py --glove-sn WG1JA02260331005 --seconds 20 --out rec
产出：
    rec.csv    宽表：t_host, sk0_x..sk20_z(63列), ja0..ja20(21列)
    rec.jsonl  每帧一行完整 JSON（含 tactile 原始数组，便于回放/离线分析）
"""
import argparse, csv, json, time, os
from wuji_sdk import SdkManager, ConnectOptions


def _pos_list(fr):
    """hand_skeleton 帧 → [[x,y,z]*21]（position 已是 [x,y,z] list）。"""
    return [list(j.pose.position) for j in fr.joints]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove-sn", default="WG1JA02260331005")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default="rec")
    ap.add_argument("--no-tactile", action="store_true")
    args = ap.parse_args()

    mgr = SdkManager.instance()
    try:
        mgr.switch_to_default_user()
    except Exception as e:
        print("[warn] switch_to_default_user:", e)
    g = mgr.connect(sn=args.glove_sn, device_name="glove",
                    options=ConnectOptions(enable_bridge=False))
    print(f"connected {args.glove_sn} side={str(g.hand_side().get())}")

    sk = g.hand_skeleton().subscribe()
    ja = g.hand_joint_angles().subscribe()
    tac = None if args.no_tactile else g.tactile().subscribe()

    csv_f = open(args.out + ".csv", "w", newline="")
    jsonl_f = open(args.out + ".jsonl", "w")
    def flat_angles(fr):
        """hand_joint_angles → 扁平角度列表（5 指 × 每指 angles 顺序拼接，固件关节序）。"""
        out = []
        for fg in fr.fingers:
            out.extend(list(fg.angles))
        return out

    # 先探一帧确定关节数（不同手型 DoF 可能不同）
    _probe = None
    _t = time.time()
    while _probe is None and time.time() - _t < 5:
        while (x := ja.recv()) is not None: _probe = x
        time.sleep(0.005)
    n_ja = len(flat_angles(_probe)) if _probe is not None else 20
    header = (["t_host"]
              + [f"sk{i}_{a}" for i in range(21) for a in ("x", "y", "z")]
              + [f"ja{i}" for i in range(n_ja)])
    w = csv.writer(csv_f); w.writerow(header)

    n, t0 = 0, time.time()
    last_sk = last_ja = last_tac = None
    try:
        while time.time() - t0 < args.seconds:
            # 只取每路最新帧（丢掉积压 → 不累计延迟）
            while (x := sk.recv()) is not None: last_sk = x
            while (x := ja.recv()) is not None: last_ja = x
            if tac is not None:
                while (x := tac.recv()) is not None: last_tac = x
            if last_sk is None or last_ja is None:
                time.sleep(0.003); continue

            t = time.time()
            sk_pts = _pos_list(last_sk)
            ja_vals = flat_angles(last_ja)
            row = [f"{t:.6f}"] + [f"{v:.6f}" for p in sk_pts for v in p] \
                  + [f"{v:.6f}" for v in ja_vals]
            w.writerow(row)
            rec = {"t_host": t, "skeleton": sk_pts, "joint_angles": ja_vals}
            if last_tac is not None:
                try:
                    rec["tactile"] = json.loads(json.dumps(last_tac, default=lambda o: getattr(o, "__dict__", str(o))))
                except Exception:
                    rec["tactile"] = str(last_tac)
            jsonl_f.write(json.dumps(rec) + "\n")
            n += 1
            time.sleep(0.003)
    finally:
        csv_f.close(); jsonl_f.close()
        mgr.disconnect_all()
    print(f"done: {n} frames, {n/max(time.time()-t0,1e-3):.1f} Hz → {args.out}.csv / {args.out}.jsonl")


if __name__ == "__main__":
    main()
