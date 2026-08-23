#!/usr/bin/env python3
"""拿**真实策略**（LeRobot ACT）在本项目数据集上压测，而不是用 MLP 探针近似。

`tools/bench_train.py` 用三档 MLP 探针测出 IO→compute 翻转点在 13M~118M 之间，
但同参数量下 transformer 的算子构成和访存模式与 MLP 差别很大，那只是量级判断。
这里直接跑 ACT——实测 **40.24M 参数**，正好落在那个区间里，所以它到底在哪一侧
必须实测。

数据映射（本项目的 108 维 observation.state 按语义拆开，不是随便切）：
  · `observation.state` ← 后 20 维 = 真机手本体感（机器人自己的状态）
  · `observation.environment_state` ← 前 88 维 = 手套骨架 63 + 手套关节 25（外部输入）
ACT 需要动作序列而不是单步，用 `delta_timestamps` 让数据集直接返回 chunk。

用法：
    ./venv312/bin/python tools/bench_policy.py --dataset data/datasets/finger_tap \\
        --repo-id local/wuji_finger_tap
    ./venv312/bin/python tools/bench_policy.py --dataset <ds> --chunk 100 \\
        --batch-sizes 8 16 --repeats 3 --json-out /tmp/act.json
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.bench_train import GpuSampler  # noqa: E402

ENV_DIM, STATE_DIM = 88, 20             # 88 = skeleton63 + glove_joints25；20 = 真机手


def _slice_stats(stats, lo, hi, full=108):
    return {k: (v[lo:hi] if hasattr(v, "__len__") and len(v) == full else v)
            for k, v in stats.items()}


def build_act(ds, chunk, device):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    S = ds.meta.stats["observation.state"]
    stats = {"observation.environment_state": _slice_stats(S, 0, ENV_DIM),
             "observation.state": _slice_stats(S, ENV_DIM, ENV_DIM + STATE_DIM),
             "action": ds.meta.stats["action"]}
    cfg = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE,
                                               shape=(STATE_DIM,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV,
                                                           shape=(ENV_DIM,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION,
                                                 shape=(20,))},
        chunk_size=chunk, n_action_steps=chunk, device=str(device))
    return ACTPolicy(cfg, dataset_stats=stats).to(device)


def to_batch(b, device):
    x = b["observation.state"]
    out = {"observation.environment_state": x[:, :ENV_DIM].to(device, non_blocking=True),
           "observation.state": x[:, ENV_DIM:].to(device, non_blocking=True),
           "action": b["action"].to(device, non_blocking=True)}
    if "action_is_pad" in b:
        out["action_is_pad"] = b["action_is_pad"].to(device, non_blocking=True)
    return out


def run_one(ds, chunk, device, batch_size, workers, amp_dtype, steps, fused_adam):
    from torch.utils.data import DataLoader

    pol = build_act(ds, chunk, device)
    n_params = sum(p.numel() for p in pol.parameters())
    opt = torch.optim.Adam(pol.parameters(), lr=1e-5,
                           **({"fused": True} if fused_adam else {}))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    kw = {"prefetch_factor": 2, "persistent_workers": True} if workers else {}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                    drop_last=True, **kw)

    torch.cuda.reset_peak_memory_stats()
    it = iter(dl)

    def step():
        nonlocal it
        t0 = time.perf_counter()
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            b = next(it)
        batch = to_batch(b, device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, _ = pol.forward(batch)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return (t1 - t0, t2 - t1, t3 - t2, time.perf_counter() - t3,
                batch["action"].shape[0])

    for _ in range(3):
        step()
    sampler = GpuSampler()
    sampler.start()
    td = tf = tb = to = 0.0
    n = 0
    t_all = time.perf_counter()
    for _ in range(steps):
        a, b_, c, d, bs = step()
        td += a
        tf += b_
        tb += c
        to += d
        n += bs
    total = time.perf_counter() - t_all
    util = sampler.stop()
    peak = torch.cuda.max_memory_allocated() / 1e6
    del it, dl, pol, opt
    torch.cuda.empty_cache()

    return {"params_m": round(n_params / 1e6, 2), "batch_size": batch_size,
            "num_workers": workers, "chunk": chunk,
            "amp": None if amp_dtype is None else str(amp_dtype).replace("torch.", ""),
            "fused_adam": fused_adam,
            "samples_per_s": round(n / total, 1), "seconds": round(total, 3),
            "pct_dataload": round(100 * td / total, 1),
            "pct_forward": round(100 * tf / total, 1),
            "pct_backward": round(100 * tb / total, 1),
            "pct_optimizer": round(100 * to / total, 1),
            "gpu_util_pct": round(util, 1) if util is not None else None,
            "peak_gpu_mb": round(peak, 1)}


def repeated(repeats, *a):
    rs = [run_one(*a) for _ in range(max(repeats, 1))]
    sp = sorted(r["samples_per_s"] for r in rs)
    out = dict(rs[0])
    out.update(samples_per_s=round(statistics.median(sp), 1),
               samples_per_s_min=sp[0], samples_per_s_max=sp[-1], repeats=len(rs),
               spread_pct=round(100 * (sp[-1] - sp[0]) / max(statistics.median(sp), 1e-9), 1))
    for k in ("pct_dataload", "pct_forward", "pct_backward", "pct_optimizer",
              "gpu_util_pct", "peak_gpu_mb"):
        vals = [r[k] for r in rs if r[k] is not None]
        if vals:
            out[k] = round(statistics.median(vals), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="真实策略（ACT）训练压测")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--repo-id", default="local/wuji_finger_tap")
    ap.add_argument("--chunk", type=int, default=100, help="ACT 动作块长度")
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[8, 16])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--baseline-workers", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability(0)
    amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    print("GPU %s  capability %d.%d  AMP=%s"
          % (torch.cuda.get_device_name(0), cap[0], cap[1],
             str(amp_dtype).replace("torch.", "")))

    base = LeRobotDataset(repo_id=args.repo_id, root=args.dataset)
    dt = {"action": [i / base.meta.fps for i in range(args.chunk)]}
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset, delta_timestamps=dt)
    print("数据集 %d 帧  fps=%d  动作块=%d（每样本 %d 步动作）\n"
          % (len(ds), base.meta.fps, args.chunk, args.chunk))

    runs = []
    print("%-10s %5s %4s %7s %10s %6s %6s %7s %7s %7s %9s"
          % ("tag", "bs", "w", "amp", "samples/s", "±%", "GPU%", "data%",
             "fwd%", "bwd%", "mem MB"))

    def show(tag, r):
        print("%-10s %5d %4d %7s %10.1f %5.0f%% %6s %7.1f %7.1f %7.1f %9.1f"
              % (tag, r["batch_size"], r["num_workers"],
                 (r["amp"] or "off") + ("+f" if r["fused_adam"] else ""),
                 r["samples_per_s"], r.get("spread_pct", 0),
                 r["gpu_util_pct"] if r["gpu_util_pct"] is not None else "-",
                 r["pct_dataload"], r["pct_forward"], r["pct_backward"],
                 r["peak_gpu_mb"]))

    r = repeated(args.repeats, ds, args.chunk, device, args.batch_sizes[0],
                 args.baseline_workers, None, args.steps, False)
    r["tag"] = "baseline"
    runs.append(r)
    show("baseline", r)

    for bs in args.batch_sizes:
        for amp in (None, amp_dtype):
            r = repeated(args.repeats, ds, args.chunk, device, bs, args.workers,
                         amp, args.steps, False)
            r["tag"] = "tuned"
            runs.append(r)
            show("tuned", r)
    r = repeated(args.repeats, ds, args.chunk, device, args.batch_sizes[-1],
                 args.workers, amp_dtype, args.steps, True)
    r["tag"] = "fused"
    runs.append(r)
    show("fused", r)

    base_r = runs[0]
    best = max(runs, key=lambda x: x["samples_per_s"])
    bound = "IO" if best["pct_dataload"] > 50 else "compute"
    print("\n=== 结论 ===")
    print("ACT %.2fM 参数：%.0f → %.0f samples/s (%.2f×)  最优配置瓶颈=%s"
          % (base_r["params_m"], base_r["samples_per_s"], best["samples_per_s"],
             best["samples_per_s"] / max(base_r["samples_per_s"], 1e-9), bound))
    print("  GPU 利用率 %s%% → %s%%   峰值显存 %.0f MB   分段 data/fwd/bwd/opt = "
          "%.0f/%.0f/%.0f/%.0f%%"
          % (base_r["gpu_util_pct"], best["gpu_util_pct"], best["peak_gpu_mb"],
             best["pct_dataload"], best["pct_forward"], best["pct_backward"],
             best["pct_optimizer"]))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(0), "runs": runs}, f,
                      ensure_ascii=False, indent=2)
        print("\n结果 → %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
