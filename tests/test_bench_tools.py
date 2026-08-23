#!/usr/bin/env python3
"""压测工具的纯逻辑单测（不碰 GPU、不读数据集，CI 里能跑）。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "需要 torch")
class BenchTrainTest(unittest.TestCase):
    def test_model_sizes_span_the_crossover(self):
        """三档规模必须真的拉开量级，否则测不出 IO→compute 翻转点。"""
        from tools.bench_train import build_model
        n = {}
        for k in ("mlp", "big", "xl"):
            n[k] = sum(p.numel() for p in build_model(k, 108, 20).parameters())
        self.assertLess(n["mlp"], 1e6)              # baseline 量级
        self.assertGreater(n["big"], 5e6)
        self.assertGreater(n["xl"], 50e6)           # 足够把 GPU 压到 compute bound
        self.assertLess(n["mlp"], n["big"] / 10)

    def test_unknown_model_rejected(self):
        from tools.bench_train import build_model
        with self.assertRaises(SystemExit):
            build_model("nope", 10, 2)

    def test_gpu_sampler_stops_cleanly(self):
        """曾经因为 self._stop 覆盖 Thread._stop 导致 join() 抛 TypeError。"""
        from tools.bench_train import GpuSampler
        s = GpuSampler(interval=0.001)
        s.start()
        s.stop()                                     # 不能抛
        self.assertFalse(s.is_alive())


@unittest.skipUnless(HAVE_TORCH, "需要 torch")
class BenchInferTest(unittest.TestCase):
    def test_percentiles(self):
        from tools.bench_infer import percentiles
        st = percentiles(list(range(1, 101)))
        self.assertEqual(st["p50"], 51)
        self.assertEqual(st["max"], 100)
        self.assertLessEqual(st["p50"], st["p95"])
        self.assertLessEqual(st["p95"], st["p99"])
        self.assertLessEqual(st["p99"], st["max"])

    def test_percentiles_single_sample(self):
        from tools.bench_infer import percentiles
        st = percentiles([3.0])
        self.assertEqual(st["p50"], 3.0)
        self.assertEqual(st["p99"], 3.0)

    def test_frame_budget_matches_120hz(self):
        """帧预算写死过会和实际采集频率脱节 —— 锁死在 120Hz。"""
        from tools.bench_infer import E2E_LATENCY_MS, FRAME_BUDGET_MS
        self.assertAlmostEqual(FRAME_BUDGET_MS, 1000.0 / 120.0, places=6)
        self.assertGreater(E2E_LATENCY_MS, FRAME_BUDGET_MS)   # 端到端必然大于单帧


@unittest.skipUnless(HAVE_TORCH, "需要 torch")
class BenchPolicyTest(unittest.TestCase):
    """ACT 压测里那个 108 维观测的语义拆分必须对得上，切错就是喂错数据。"""

    def test_dims_split_matches_export(self):
        from tools.bench_policy import ENV_DIM, STATE_DIM
        # 导出用的是 --obs skeleton,joints,hand = 63 + 25 + 20 = 108
        self.assertEqual(ENV_DIM + STATE_DIM, 108)
        self.assertEqual(STATE_DIM, 20)          # 真机手本体感 = 20 关节
        self.assertEqual(ENV_DIM, 63 + 25)       # 手套骨架 21×3 + 手套关节 25

    def test_slice_stats_only_touches_full_length_vectors(self):
        """stats 里既有 108 维向量也有标量，切片不能把标量也切了。"""
        import numpy as np
        from tools.bench_policy import _slice_stats
        st = {"mean": np.arange(108.0), "std": np.ones(108), "count": np.array([5])}
        out = _slice_stats(st, 0, 88)
        self.assertEqual(len(out["mean"]), 88)
        self.assertEqual(len(out["std"]), 88)
        self.assertEqual(len(out["count"]), 1)   # 标量原样保留
        self.assertEqual(out["mean"][0], 0.0)
        out2 = _slice_stats(st, 88, 108)
        self.assertEqual(out2["mean"][0], 88.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
