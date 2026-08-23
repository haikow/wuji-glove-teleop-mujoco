#!/usr/bin/env python3
"""训练吞吐 / 显存 / GPU 利用率 profile —— 回答"训练侧瓶颈在 IO 还是 compute"。

`tools/bench_pipeline.py` 只测到 dataloader 出口；这里把真实训练步接上，
按 **dataload / forward / backward / optimizer** 分段计时，并采样 GPU 利用率与显存。

为什么要扫模型规模：BC baseline 那个 MLP 只有几万参数，在任何现代 GPU 上都是空转，
测出来只会是"GPU 利用率个位数"。扫到大模型才能定出**瓶颈从 IO 转到 compute 的临界点**，
那个临界点才是选 batch / worker / 是否上 AMP 的依据。

AMP 精度选择：Turing（compute 7.5，如 RTX 2060）**没有原生 bf16 张量核**，
`torch.cuda.is_bf16_supported()` 返回 True 但走的是模拟路径，会更慢 —— 这类卡要用 fp16。
本脚本默认按 capability 自动选，`--amp-dtype` 可强制。

用法：
    ./venv312/bin/python tools/bench_train.py --dataset data/datasets/finger_tap \\
        --repo-id local/wuji_finger_tap
    ./venv312/bin/python tools/bench_train.py --dataset <ds> --models mlp big xl \\
        --amp both --json-out /tmp/train_bench.json
"""
import argparse
import json
import os
import sys
import statistics
import threading
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_model(kind, obs_dim, action_dim):
    """三档规模，用来找 IO→compute 的临界点。"""
    if kind == "mlp":                                   # BC baseline 同款
        hidden, layers = 256, 2
    elif kind == "big":
        hidden, layers = 2048, 4
    elif kind == "xl":
        hidden, layers = 4096, 8
    else:
        raise SystemExit("未知模型档位：%s" % kind)
    mods, d = [], obs_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, action_dim))
    return nn.Sequential(*mods)


class GpuSampler(threading.Thread):
    """后台采样 GPU 利用率。torch.cuda.utilization() 是瞬时值，必须多点平均。"""

    def __init__(self, interval=0.02):
        super().__init__(daemon=True)
        # 不能叫 self._stop —— 会覆盖 threading.Thread._stop 方法，join() 时炸
        self.interval, self.samples = interval, []
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.samples.append(torch.cuda.utilization())
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=1.0)
        return (sum(self.samples) / len(self.samples)) if self.samples else None


def run_repeated(repeats, *a, **kw):
    """重复跑取中位数并带上区间。

    IO 绑死的配置对系统状态（页缓存、CPU 争用）非常敏感，实测单次之间能差 40%——
    只报单次会得出错误的提速倍数。这里默认取中位数并把 min/max 一起报出来。
    """
    rs = [run_one(*a, **kw) for _ in range(max(repeats, 1))]
    sp = sorted(r["samples_per_s"] for r in rs)
    base = dict(rs[0])
    base.update(samples_per_s=round(statistics.median(sp), 1),
                samples_per_s_min=sp[0], samples_per_s_max=sp[-1],
                repeats=len(rs),
                spread_pct=round(100 * (sp[-1] - sp[0]) / max(statistics.median(sp), 1e-9), 1))
    for k in ("pct_dataload", "pct_forward", "pct_backward", "pct_optimizer",
              "gpu_util_pct", "peak_gpu_mb"):
        vals = [r[k] for r in rs if r[k] is not None]
        if vals:
            base[k] = round(statistics.median(vals), 1)
    return base


def run_one(ds, model_kind, obs_dim, action_dim, device, batch_size, workers,
            amp_dtype, steps, compile_model=False, fused_adam=False):
    """跑 steps 个训练步，返回分段耗时与吞吐。"""
    from torch.utils.data import DataLoader

    net = build_model(model_kind, obs_dim, action_dim).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    if compile_model:
        net = torch.compile(net)
    # 大模型上 Adam 的 step 会变成大头（xl 118M 参数时占 43.9%），
    # fused 版本把逐张量的 elementwise kernel 合成一个，能明显削掉这块。
    opt = torch.optim.Adam(net.parameters(), lr=1e-3,
                           **({"fused": True} if fused_adam and device.type == "cuda"
                              else {}))
    lossf = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    kw = {"prefetch_factor": 2, "persistent_workers": True} if workers else {}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                    drop_last=True, **kw)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    it = iter(dl)
    # 预热：worker 启动 + cudnn autotune + （如有）compile
    for _ in range(3):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            b = next(it)
        x = b["observation.state"].to(device, non_blocking=True)
        y = b["action"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = lossf(net(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    sampler = None
    if device.type == "cuda":
        sampler = GpuSampler()
        sampler.start()

    t_data = t_fwd = t_bwd = t_opt = 0.0
    n = 0
    t_all = time.perf_counter()
    for _ in range(steps):
        t0 = time.perf_counter()
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            b = next(it)
        x = b["observation.state"].to(device, non_blocking=True)
        y = b["action"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_data += t1 - t0

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = lossf(net(x), y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_fwd += t2 - t1

        scaler.scale(loss).backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        t_bwd += t3 - t2

        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_opt += time.perf_counter() - t3
        n += x.shape[0]
    total = time.perf_counter() - t_all

    util = sampler.stop() if sampler else None
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else None
    del it, dl, net, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_kind, "params_m": round(n_params / 1e6, 3),
        "batch_size": batch_size, "num_workers": workers,
        "amp": None if amp_dtype is None else str(amp_dtype).replace("torch.", ""),
        "compiled": compile_model, "fused_adam": fused_adam,
        "samples": n, "seconds": round(total, 3),
        "samples_per_s": round(n / total, 1),
        "pct_dataload": round(100 * t_data / total, 1),
        "pct_forward": round(100 * t_fwd / total, 1),
        "pct_backward": round(100 * t_bwd / total, 1),
        "pct_optimizer": round(100 * t_opt / total, 1),
        "gpu_util_pct": round(util, 1) if util is not None else None,
        "peak_gpu_mb": round(peak_mb, 1) if peak_mb is not None else None,
    }


def main():
    ap = argparse.ArgumentParser(description="训练吞吐 / 显存 / GPU 利用率 profile")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--repo-id", default="local/wuji_finger_tap")
    ap.add_argument("--models", nargs="*", default=["mlp", "big", "xl"],
                    choices=["mlp", "big", "xl"])
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[256, 1024])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--baseline-workers", type=int, default=0,
                    help="朴素基线的 num_workers（对照组）")
    ap.add_argument("--amp", default="both", choices=["off", "on", "both"])
    ap.add_argument("--amp-dtype", default="auto", choices=["auto", "fp16", "bf16"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=3,
                    help="每个配置重复几次取中位数。IO 绑死的配置单次波动可达 ±40%%，"
                         "单次数字不可信")
    ap.add_argument("--compile", action="store_true", help="额外测一档 torch.compile")
    ap.add_argument("--fused-adam", action="store_true",
                    help="额外测一档 fused Adam（大模型上 optimizer 占比高时有效）")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("没有可用 CUDA —— 这个脚本就是来测 GPU 侧的")
    cap = torch.cuda.get_device_capability(0)
    # Turing(7.5) 的 is_bf16_supported() 返回 True 但没有原生张量核，实际更慢
    auto_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    amp_dtype = {"auto": auto_dtype, "fp16": torch.float16,
                 "bf16": torch.bfloat16}[args.amp_dtype]
    print("GPU: %s  capability=%d.%d  AMP dtype=%s"
          % (torch.cuda.get_device_name(0), cap[0], cap[1],
             str(amp_dtype).replace("torch.", "")))
    if args.amp_dtype == "auto" and cap[0] < 8:
        print("  （Turing 及更早没有原生 bf16 张量核，自动选 fp16）")

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset)
    s0 = ds[0]
    obs_dim = int(s0["observation.state"].shape[-1])
    action_dim = int(s0["action"].shape[-1])
    print("数据集 %d 帧  obs=%d  action=%d\n" % (len(ds), obs_dim, action_dim))

    amp_modes = {"off": [None], "on": [amp_dtype], "both": [None, amp_dtype]}[args.amp]
    runs = []

    print("每个配置重复 %d 次取中位数（IO 绑死的配置单次波动可达 ±40%%）\n" % args.repeats)
    print("%-6s %8s %6s %4s %6s %10s %6s %6s %7s %7s %7s %8s"
          % ("model", "params M", "bs", "w", "amp", "samples/s", "±%", "GPU%",
             "data%", "fwd%", "bwd%", "mem MB"))

    def show(r):
        print("%-6s %8.3f %6d %4d %6s %10.1f %5.0f%% %6s %7.1f %7.1f %7.1f %8.1f"
              % (r["model"], r["params_m"], r["batch_size"], r["num_workers"],
                 (r["amp"] or "off") + ("+f" if r.get("fused_adam") else ""),
                 r["samples_per_s"], r.get("spread_pct", 0),
                 r["gpu_util_pct"] if r["gpu_util_pct"] is not None else "-",
                 r["pct_dataload"], r["pct_forward"], r["pct_backward"],
                 r["peak_gpu_mb"]))

    for mk in args.models:
        # 朴素基线：num_workers=0、无 AMP、小 batch —— 很多人默认就这么写
        r = run_repeated(args.repeats, ds, mk, obs_dim, action_dim, device,
                         args.batch_sizes[0], args.baseline_workers, None, args.steps)
        r["tag"] = "baseline"
        runs.append(r)
        show(r)
        for bs in args.batch_sizes:
            for amp in amp_modes:
                r = run_repeated(args.repeats, ds, mk, obs_dim, action_dim, device,
                                 bs, args.workers, amp, args.steps)
                r["tag"] = "tuned"
                runs.append(r)
                show(r)
        if args.fused_adam:
            r = run_repeated(args.repeats, ds, mk, obs_dim, action_dim, device,
                             args.batch_sizes[-1], args.workers, amp_modes[-1],
                             args.steps, fused_adam=True)
            r["tag"] = "fused_adam"
            runs.append(r)
            show(r)
        if args.compile:
            r = run_repeated(args.repeats, ds, mk, obs_dim, action_dim, device,
                             args.batch_sizes[-1], args.workers, amp_modes[-1],
                             args.steps, compile_model=True)
            r["tag"] = "compiled"
            runs.append(r)
            show(r)
        print()

    print("=== 结论 ===")
    for mk in args.models:
        rs = [r for r in runs if r["model"] == mk]
        base = next(r for r in rs if r["tag"] == "baseline")
        best = max(rs, key=lambda r: r["samples_per_s"])
        bound = "IO" if best["pct_dataload"] > 50 else "compute"
        print("%-5s(%.2fM 参数)  %.0f → %.0f samples/s (%.2f×)  最优配置瓶颈=%s  "
              "GPU 利用率 %s%% → %s%%"
              % (mk, base["params_m"], base["samples_per_s"], best["samples_per_s"],
                 best["samples_per_s"] / max(base["samples_per_s"], 1e-9), bound,
                 base["gpu_util_pct"], best["gpu_util_pct"]))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(0),
                       "capability": list(cap), "runs": runs}, f,
                      ensure_ascii=False, indent=2)
        print("\n结果 → %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
