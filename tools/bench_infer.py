#!/usr/bin/env python3
"""策略推理延迟 —— 和遥操的时间预算对账，回答"这个策略能不能塞进回路"。

遥操回路里每一帧的预算是硬的：

  · 手套出帧 **120Hz → 8.33ms/帧**。策略推理必须显著小于它，否则会掉帧。
  · 端到端（手套出帧 → retarget → zenoh → 伺服 → 反馈）实测 **33ms**
    （见 docs/findings.md §4）。策略插进回路后延迟会叠加在这 33ms 上。

所以这里只测 **batch=1 的尾延迟**（p50/p95/p99/max），不测吞吐 —— 遥操是逐帧同步调用，
平均值没有意义，决定掉不掉帧的是 p99。同时对比 CPU / GPU：小模型在 GPU 上常常更慢，
因为一次 H2D + kernel launch 的固定开销盖过了计算。

用法：
    ./venv312/bin/python tools/bench_infer.py --model data/models/bc_tap.pt
    ./venv312/bin/python tools/bench_infer.py --model <pt> --batch-sizes 1 8 32 \\
        --json-out /tmp/infer.json
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.train_bc import MLP  # noqa: E402

FRAME_BUDGET_MS = 1000.0 / 120.0        # 手套 120Hz
E2E_LATENCY_MS = 33.0                   # 实测端到端（docs/findings.md §4）


def percentiles(xs):
    s = sorted(xs)
    def q(p):
        return s[min(int(round(p * (len(s) - 1))), len(s) - 1)]
    return {"p50": q(0.50), "p90": q(0.90), "p95": q(0.95), "p99": q(0.99),
            "max": s[-1], "mean": statistics.fmean(s)}


def bench(net, obs_dim, device, batch, iters, warmup, use_amp=False):
    """逐次同步计时。GPU 上必须 synchronize，否则测到的是入队时间不是完成时间。"""
    net = net.to(device).eval()
    x = torch.randn(batch, obs_dim, device=device)
    amp_kw = {"device_type": device.type, "dtype": torch.float16,
              "enabled": use_amp and device.type == "cuda"}
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(**amp_kw):
                net(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat = []
        for _ in range(iters):
            t0 = time.perf_counter()
            with torch.autocast(**amp_kw):
                net(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
    return lat


def bench_e2e_single(net, obs_dim, device, iters, warmup):
    """把 numpy→tensor→H2D→forward→D2H→numpy 全算进去。

    真实回路里输入来自手套（numpy），输出要发给手（numpy），只测 forward 会低估。
    """
    net = net.to(device).eval()
    obs = np.random.randn(obs_dim).astype(np.float32)
    with torch.no_grad():
        for _ in range(warmup):
            t = torch.from_numpy(obs).unsqueeze(0).to(device)
            net(t).squeeze(0).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat = []
        for _ in range(iters):
            t0 = time.perf_counter()
            t = torch.from_numpy(obs).unsqueeze(0).to(device)
            out = net(t).squeeze(0).cpu().numpy()
            lat.append((time.perf_counter() - t0) * 1000.0)
        assert out.shape[0] > 0
    return lat


def main():
    ap = argparse.ArgumentParser(description="策略推理延迟 vs 遥操时间预算")
    ap.add_argument("--model", required=True, help="tools/train_bc.py 存的 .pt")
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 8, 32])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU 推理线程数；0=不改。遥操场景常要限成 1~2 避免抢核")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    obs_dim, action_dim = ck["obs_dim"], ck["action_dim"]
    net = MLP(obs_dim, action_dim, ck.get("hidden", 256), ck.get("layers", 2))
    net.load_state_dict(ck["state_dict"])
    n_params = sum(p.numel() for p in net.parameters())
    print("模型 %s  obs=%d action=%d  %.3fM 参数  CPU 线程=%d"
          % (os.path.basename(args.model), obs_dim, action_dim, n_params / 1e6,
             torch.get_num_threads()))
    print("预算：手套 120Hz → 每帧 %.2fms；端到端实测 %.0fms\n"
          % (FRAME_BUDGET_MS, E2E_LATENCY_MS))

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    runs = []
    print("%-6s %5s %6s %8s %8s %8s %8s %8s  %s"
          % ("device", "bs", "amp", "p50 ms", "p90 ms", "p95 ms", "p99 ms",
             "max ms", "占帧预算(p99)"))
    for dev in devices:
        for bs in args.batch_sizes:
            for amp in ([False, True] if dev.type == "cuda" else [False]):
                lat = bench(net, obs_dim, dev, bs, args.iters, args.warmup, amp)
                st = percentiles(lat)
                rec = {"device": dev.type, "batch_size": bs, "amp_fp16": amp,
                       "kind": "forward", **{k: round(v, 4) for k, v in st.items()}}
                runs.append(rec)
                print("%-6s %5d %6s %8.3f %8.3f %8.3f %8.3f %8.3f  %6.1f%%"
                      % (dev.type, bs, "fp16" if amp else "-", st["p50"], st["p90"],
                         st["p95"], st["p99"], st["max"],
                         100 * st["p99"] / FRAME_BUDGET_MS))

    print("\n=== 端到端单帧（numpy → tensor → H2D → forward → D2H → numpy）===")
    print("%-6s %8s %8s %8s %8s  %s"
          % ("device", "p50 ms", "p95 ms", "p99 ms", "max ms", "占帧预算(p99)"))
    best = None
    for dev in devices:
        lat = bench_e2e_single(net, obs_dim, dev, args.iters, args.warmup)
        st = percentiles(lat)
        rec = {"device": dev.type, "batch_size": 1, "kind": "end_to_end",
               **{k: round(v, 4) for k, v in st.items()}}
        runs.append(rec)
        print("%-6s %8.3f %8.3f %8.3f %8.3f  %6.1f%%"
              % (dev.type, st["p50"], st["p95"], st["p99"], st["max"],
                 100 * st["p99"] / FRAME_BUDGET_MS))
        if best is None or st["p99"] < best[1]["p99"]:
            best = (dev.type, st)

    dev_name, st = best
    head = FRAME_BUDGET_MS / st["p99"]
    print("\n=== 结论 ===")
    print("最优部署：%s，单帧端到端 p99 = %.3f ms" % (dev_name, st["p99"]))
    print("  占 120Hz 帧预算 %.2fms 的 %.1f%%，余量 %.1f×"
          % (FRAME_BUDGET_MS, 100 * st["p99"] / FRAME_BUDGET_MS, head))
    print("  叠加到已实测的 %.0fms 端到端遥操延迟上 → %.2f ms（+%.1f%%）"
          % (E2E_LATENCY_MS, E2E_LATENCY_MS + st["p99"],
             100 * st["p99"] / E2E_LATENCY_MS))
    print("  判定：%s" % ("可以塞进遥操回路" if st["p99"] < FRAME_BUDGET_MS * 0.5
                          else "余量不足，需要裁剪模型或异步化"))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"model": os.path.abspath(args.model), "params": n_params,
                       "frame_budget_ms": FRAME_BUDGET_MS,
                       "e2e_latency_ms": E2E_LATENCY_MS, "runs": runs}, f,
                      ensure_ascii=False, indent=2)
        print("\n结果 → %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
