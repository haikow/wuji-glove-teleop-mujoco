#!/usr/bin/env python3
"""数据链路规模压测：合成 N 条 episode，逐段测吞吐 / 耗时 / 峰值内存 / 磁盘占用。

真机采集受人力限制（一条 10~15s 还要复位），拿它测不出管线的规模边界。这里用合成
episode 把 QC / 导出 / dataloader 三段各自压到已知量级，回答"这条链路能撑多少数据、
瓶颈在哪一段"。

分段口径
  gen        合成 episode（JSONL 容器）—— 只是造数据，不算管线性能
  qc         tools/qc_episode.py 的 compute_metrics + evaluate（纯标准库，单进程）
  export     LeRobotDataset.create/add_frame/save_episode（写 parquet + meta 分片）
  dataload   LeRobotDataset 随机读，模拟训练时的取样吞吐（最接近训练侧瓶颈的指标）

用法：
    ./venv312/bin/python tools/bench_pipeline.py --episodes 100 --frames 1200
    ./venv312/bin/python tools/bench_pipeline.py --episodes 1000 --frames 300 --skip dataload
    ./venv312/bin/python tools/bench_pipeline.py --episodes 50 --dataloader-workers 0 2 4 8
"""
import argparse
import json
import os
import resource
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import synth_episode  # noqa: E402
from tools.export_dataset import (  # noqa: E402
    _obs_names, _obs_vector, parse_obs_mode, skip_embed_images_when_no_media,
)
from tools.qc_episode import qc_episode  # noqa: E402


def peak_rss_mb():
    """ru_maxrss 在 Linux 上是 KB。这是整个进程的峰值，不是分段增量。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def dir_size_mb(path):
    tot = 0
    for root, _d, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot / 1e6


class Stage:
    """计时 + 记录峰值内存的小上下文。"""

    def __init__(self, name, results):
        self.name, self.results = name, results

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0
        self.results[self.name] = {"seconds": round(self.dt, 3),
                                   "peak_rss_mb": round(peak_rss_mb(), 1)}
        return False


def bench_generate(root, n, frames, hz):
    paths = []
    for i in range(n):
        paths.append(synth_episode.make(root, "none", n=frames, hz=hz,
                                        task="bench", episode_id="ep_bench_%05d" % i))
    return paths


def bench_qc(paths):
    npass = 0
    for p in paths:
        npass += bool(qc_episode(p)["pass"])
    return npass


def bench_export(input_dir, out, repo_id, obs_mode, fps, skip_embed=True):
    """绕过 CLI 直接调库，免得把 argparse 和打印算进耗时。

    skip_embed 与 tools/export_dataset.py 走同一个开关，压测数字才和实际导出一致。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from tools.episode_format import iter_frames, load_meta

    eps = sorted(os.path.join(input_dir, d) for d in os.listdir(input_dir)
                 if d.startswith("ep_bench_"))
    first = next(iter_frames(eps[0]))
    odim = len(_obs_vector(first, obs_mode))
    adim = len(first["action"])
    n_ja = len(first.get("joint_angles") or [])
    features = {
        "observation.state": {"dtype": "float32", "shape": (odim,),
                              "names": _obs_names(obs_mode, n_ja)},
        "action": {"dtype": "float32", "shape": (adim,), "names": None},
    }
    import numpy as np
    total = 0
    with skip_embed_images_when_no_media(skip_embed):
        ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features,
                                   root=out, robot_type="bench", use_videos=False)
        for ep in eps:
            meta = load_meta(ep)
            for f in iter_frames(ep):
                ds.add_frame({
                    "observation.state": np.asarray(_obs_vector(f, obs_mode), np.float32),
                    "action": np.asarray(f["action"], np.float32),
                    "task": meta.get("task") or "bench",
                })
                total += 1
            ds.save_episode()
    return total, odim, adim


def _one_dataload(ds, w, bs, max_batches, pin, prefetch):
    from torch.utils.data import DataLoader

    kw = {}
    if w:                                           # prefetch_factor 只在有 worker 时合法
        kw["prefetch_factor"] = prefetch
        kw["persistent_workers"] = True
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=w,
                    pin_memory=pin, **kw)
    it = iter(dl)
    next(it)                                        # 预热：第一批含 worker 启动开销
    t0 = time.perf_counter()
    n = 0
    for _ in range(max_batches):
        try:
            b = next(it)
        except StopIteration:
            break
        n += b["action"].shape[0]
    dt = time.perf_counter() - t0
    del it, dl
    return n, dt


def bench_dataload(root, repo_id, workers_list, batch_size, max_batches,
                   sweep=False, pin=False, prefetch=2):
    """随机读吞吐 —— 训练时真正的瓶颈指标，比"导出多快"更值钱。

    sweep=True 时额外扫 batch_size × pin_memory，用来找"朴素默认配置"到"调过参"
    之间的差距（很多人直接用 num_workers=0 的默认值，那是最慢的一档）。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=repo_id, root=root)
    out = []
    combos = [(w, batch_size, pin, prefetch) for w in workers_list]
    if sweep:
        best_w = max(workers_list)
        combos += [(best_w, bs, p, pf)
                   for bs in (batch_size * 2, batch_size * 4)
                   for p in (False, True)
                   for pf in (2, 4)]
    for w, bs, p, pf in combos:
        n, dt = _one_dataload(ds, w, bs, max_batches, p, pf)
        rec = {"num_workers": w, "batch_size": bs, "pin_memory": p,
               "prefetch_factor": pf if w else None, "samples": n,
               "seconds": round(dt, 3),
               "samples_per_s": round(n / dt, 1) if dt > 0 else None}
        out.append(rec)
        print("    w=%-2d bs=%-4d pin=%-5s pf=%-4s  %9.1f samples/s"
              % (w, bs, p, rec["prefetch_factor"], rec["samples_per_s"]))
    del ds
    return out


def main():
    ap = argparse.ArgumentParser(description="数据链路规模压测")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--frames", type=int, default=1200, help="每条 episode 帧数")
    ap.add_argument("--hz", type=float, default=120.0)
    ap.add_argument("--obs", default="both")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--dataloader-workers", type=int, nargs="*", default=[0, 4],
                    help="要对比的 num_workers 列表")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["qc", "export", "dataload"])
    ap.add_argument("--workdir", default="", help="留空用临时目录，跑完删除")
    ap.add_argument("--dataset", default="",
                    help="只跑 dataloader 压测：直接吃已有的 LeRobot 数据集目录")
    ap.add_argument("--sweep", action="store_true",
                    help="dataloader 额外扫 batch_size × pin_memory × prefetch")
    ap.add_argument("--repo-id", default="local/bench")
    ap.add_argument("--no-skip-embed", action="store_true",
                    help="导出走官方原始路径（不跳过 embed_images），用于前后对比")
    ap.add_argument("--json-out", default="", help="把结果写成 JSON")
    args = ap.parse_args()

    parse_obs_mode(args.obs)                        # 早失败

    if args.dataset:                                # 只测取样吞吐，不重新造数据
        print("dataloader 压测：%s" % args.dataset)
        runs = bench_dataload(args.dataset, args.repo_id, args.dataloader_workers,
                              args.batch_size, args.max_batches, args.sweep)
        best = max(runs, key=lambda r: r["samples_per_s"] or 0)
        base = next((r for r in runs if r["num_workers"] == 0), None)
        print("\n最快：%s" % json.dumps(best, ensure_ascii=False))
        if base and base["samples_per_s"]:
            print("相对 num_workers=0 的朴素默认：%.2f×"
                  % (best["samples_per_s"] / base["samples_per_s"]))
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"dataload_runs": runs, "best": best}, f,
                          ensure_ascii=False, indent=2)
            print("结果 → %s" % args.json_out)
        return 0

    tmp = args.workdir or tempfile.mkdtemp(prefix="bench_")
    eps_dir = os.path.join(tmp, "episodes")
    ds_dir = os.path.join(tmp, "dataset")
    results = {"config": vars(args), "host": {"cpu_count": os.cpu_count()}}
    n_frames_total = args.episodes * args.frames
    print("压测：%d 条 × %d 帧 = %d 帧，工作目录 %s"
          % (args.episodes, args.frames, n_frames_total, tmp))

    try:
        print("[1/4] 合成 episode ...")
        with Stage("gen", results) as st:
            bench_generate(eps_dir, args.episodes, args.frames, args.hz)
        results["gen"]["episodes_per_s"] = round(args.episodes / st.dt, 1)
        results["gen"]["disk_mb"] = round(dir_size_mb(eps_dir), 1)
        print("      %.1fs  %.1f 条/s  落盘 %.1f MB"
              % (st.dt, args.episodes / st.dt, results["gen"]["disk_mb"]))

        paths = sorted(os.path.join(eps_dir, d) for d in os.listdir(eps_dir))
        if "qc" not in args.skip:
            print("[2/4] QC ...")
            with Stage("qc", results) as st:
                npass = bench_qc(paths)
            results["qc"].update(episodes_per_s=round(args.episodes / st.dt, 1),
                                 frames_per_s=round(n_frames_total / st.dt, 0),
                                 passed=npass)
            print("      %.1fs  %.1f 条/s  %.0f 帧/s  通过 %d/%d"
                  % (st.dt, args.episodes / st.dt, n_frames_total / st.dt,
                     npass, args.episodes))

        if "export" not in args.skip:
            print("[3/4] 导出 LeRobot ...")
            with Stage("export", results) as st:
                total, odim, adim = bench_export(eps_dir, ds_dir, args.repo_id,
                                                 args.obs, int(args.hz),
                                                 not args.no_skip_embed)
            results["export"].update(frames=total, obs_dim=odim, action_dim=adim,
                                     frames_per_s=round(total / st.dt, 0),
                                     disk_mb=round(dir_size_mb(ds_dir), 1))
            print("      %.1fs  %.0f 帧/s  obs=%d action=%d  数据集 %.1f MB"
                  % (st.dt, total / st.dt, odim, adim, results["export"]["disk_mb"]))
            print("      压缩比：episode %.1f MB → parquet %.1f MB (%.2f×)"
                  % (results["gen"]["disk_mb"], results["export"]["disk_mb"],
                     results["export"]["disk_mb"] / max(results["gen"]["disk_mb"], 1e-9)))

        if "dataload" not in args.skip and "export" not in args.skip:
            print("[4/4] dataloader 吞吐 ...")
            with Stage("dataload", results):
                results["dataload_runs"] = bench_dataload(
                    ds_dir, args.repo_id, args.dataloader_workers,
                    args.batch_size, args.max_batches, args.sweep)

        results["peak_rss_mb"] = round(peak_rss_mb(), 1)
        print("\n峰值内存 %.1f MB" % results["peak_rss_mb"])
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print("结果 → %s" % args.json_out)
    finally:
        if not args.workdir:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
