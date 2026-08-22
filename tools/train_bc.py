#!/usr/bin/env python3
"""BC baseline：从导出的 LeRobot 数据集训一个 obs→action 的 MLP。

这不是要做 SOTA 策略，而是**飞轮的消费端** —— 它是唯一能证明"导出的数据真的能被
训练栈吃进去"的手段。QC 全绿只说明数据自洽，不说明格式对；只有跑通
LeRobotDataset → DataLoader → 反向传播 → 回放，这个闭环才算合上。

用法：
    ./venv312/bin/python tools/train_bc.py --dataset data/datasets/pick_cube \
        --repo-id local/wuji_pick_cube --epochs 20 --out data/models/bc.pt

    # 回放预测结果，肉眼看手动得对不对
    MUJOCO_GL=egl ./venv312/bin/python tools/viz_episode.py data/episodes/ep_xxx \
        --policy data/models/bc.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256, layers=2):
        super().__init__()
        mods, d = [], obs_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods.append(nn.Linear(d, action_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def load_arrays(root, repo_id):
    """LeRobotDataset → (X, Y, episode_index)，全部读进内存（这个量级够用）。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=repo_id, root=root)
    n = len(ds)
    s0 = ds[0]
    X = np.empty((n, s0["observation.state"].shape[-1]), np.float32)
    Y = np.empty((n, s0["action"].shape[-1]), np.float32)
    E = np.empty(n, np.int64)
    for i in range(n):
        s = ds[i]
        X[i] = s["observation.state"].numpy()
        Y[i] = s["action"].numpy()
        E[i] = int(s["episode_index"])
    return X, Y, E, ds.meta


def main():
    ap = argparse.ArgumentParser(description="BC baseline（飞轮消费端自检）")
    ap.add_argument("--dataset", default="data/datasets/pick_cube")
    ap.add_argument("--repo-id", default="local/wuji_pick_cube")
    ap.add_argument("--out", default="data/models/bc.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--val-episodes", type=int, default=1,
                    help="留几条 episode 做验证（按 episode 切，不按帧切）")
    ap.add_argument("--obs-mode", default="both", help="只写进 checkpoint 供回放时对齐")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, Y, E, meta = load_arrays(args.dataset, args.repo_id)
    eps = sorted(set(E.tolist()))
    print("数据集：%d 帧 / %d episode  obs=%d action=%d fps=%s"
          % (len(X), len(eps), X.shape[1], Y.shape[1], meta.fps))

    # 按 episode 切分，绝不能按帧随机切 —— 同一条 demo 的相邻帧几乎一样，
    # 帧级切分会让验证集泄漏训练集内容，误差看起来好得离谱。
    n_val = min(args.val_episodes, max(len(eps) - 1, 0))
    val_eps = set(eps[-n_val:]) if n_val else set()
    tr = ~np.isin(E, list(val_eps)) if val_eps else np.ones(len(E), bool)
    va = ~tr
    print("  train %d 帧 / %d ep   val %d 帧 / %d ep"
          % (tr.sum(), len(eps) - len(val_eps), va.sum(), len(val_eps)))
    if va.sum() == 0:
        print("  [warn] 没有验证集（episode 太少），val 指标不可信")

    om, os_ = X[tr].mean(0), X[tr].std(0) + 1e-6
    am, as_ = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    Xn = torch.from_numpy((X - om) / os_)
    Yn = torch.from_numpy((Y - am) / as_)

    dev = torch.device(args.device)
    net = MLP(X.shape[1], Y.shape[1], args.hidden, args.layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = nn.MSELoss()

    idx_tr = np.flatnonzero(tr)
    Xtr, Ytr = Xn[idx_tr].to(dev), Yn[idx_tr].to(dev)
    if va.sum():
        idx_va = np.flatnonzero(va)
        Xva, Yva = Xn[idx_va].to(dev), Yn[idx_va].to(dev)
        Yva_raw = torch.from_numpy(Y[idx_va]).to(dev)

    for ep in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(len(Xtr), device=dev)
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = lossf(net(Xtr[b]), Ytr[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        line = "epoch %3d  train_mse=%.6f" % (ep, tot / len(Xtr))
        if va.sum():
            net.eval()
            with torch.no_grad():
                pv = net(Xva)
                # 反归一化后看物理量纲的误差，比归一化 MSE 直观
                pred = pv * torch.from_numpy(as_).to(dev) + torch.from_numpy(am).to(dev)
                mae = (pred - Yva_raw).abs().mean().item()
                line += "  val_mse=%.6f  val_MAE=%.5f rad (%.3f°)" % (
                    lossf(pv, Yva).item(), mae, np.degrees(mae))
        if ep % max(args.epochs // 10, 1) == 0 or ep == 1:
            print(line)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "state_dict": net.cpu().state_dict(),
        "obs_dim": int(X.shape[1]), "action_dim": int(Y.shape[1]),
        "hidden": args.hidden, "layers": args.layers, "obs_mode": args.obs_mode,
        "obs_mean": om, "obs_std": os_, "action_mean": am, "action_std": as_,
        "dataset": os.path.abspath(args.dataset), "repo_id": args.repo_id,
    }, args.out)
    print("模型 → %s" % args.out)
    print("回放验证： MUJOCO_GL=egl ./venv312/bin/python tools/viz_episode.py "
          "<episode> --policy %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
