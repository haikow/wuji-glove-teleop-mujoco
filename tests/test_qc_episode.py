#!/usr/bin/env python3
"""QC 规则单测：合成带缺陷的 episode → 断言 QC 判定。

不需要手套/SDK/MuJoCo/numpy，纯标准库：

    python3 tests/test_qc_episode.py           # 或 python3 -m unittest discover tests
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import synth_episode  # noqa: E402
from tools.episode_format import load_frames, load_meta, quality_path  # noqa: E402
from tools.qc_episode import compute_metrics, qc_episode, route  # noqa: E402


class QCTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="qc_test_")
        self.eps = os.path.join(self.tmp, "episodes")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _qc(self, defect, **kw):
        path = synth_episode.make(self.eps, defect, **kw)
        return path, qc_episode(path)

    # ---- 正常样本必须过 ----
    def test_clean_passes(self):
        path, q = self._qc("none")
        self.assertTrue(q["pass"], q["flags"])
        self.assertEqual(q["flags"], [])
        m = q["metrics"]
        self.assertEqual(m["n_frames"], 200)
        self.assertAlmostEqual(m["rate_hz_median"], 100.0, delta=1.0)
        self.assertEqual(m["time_source"], "device")     # 有 t_dev_us 就该用设备时钟
        self.assertEqual(m["signal_key"], "action")
        self.assertEqual(m["dropout_ratio"], 0.0)
        self.assertEqual(m["dup_ratio"], 0.0)

    def test_quality_json_written(self):
        path, q = self._qc("none")
        self.assertTrue(os.path.isfile(quality_path(path)))
        with open(quality_path(path)) as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["pass"], q["pass"])
        self.assertEqual(on_disk["episode_id"], load_meta(path)["episode_id"])

    # ---- 每种缺陷都要被对应的 flag 抓到 ----
    def test_dropout_flagged(self):
        _, q = self._qc("dropout")
        self.assertIn("dropout", q["flags"])
        self.assertGreater(q["metrics"]["dropout_ratio"], 0.05)

    def test_duplicate_seq_flagged(self):
        _, q = self._qc("duplicate")
        self.assertIn("duplicate_seq", q["flags"])
        self.assertGreater(q["metrics"]["dup_frames"], 0)

    def test_static_flagged(self):
        _, q = self._qc("static")
        self.assertIn("near_static", q["flags"])

    def test_jump_flagged(self):
        _, q = self._qc("jump")
        self.assertIn("action_jump", q["flags"])
        self.assertGreater(q["metrics"]["signal_jump_max_rad"], 0.5)

    def test_short_flagged(self):
        _, q = self._qc("short")
        self.assertIn("too_short", q["flags"])

    def test_lowrate_flagged(self):
        _, q = self._qc("lowrate")
        self.assertIn("low_rate", q["flags"])

    def test_gap_flagged(self):
        _, q = self._qc("gap")
        self.assertIn("gap", q["flags"])
        self.assertGreater(q["metrics"]["dt_max_s"], 0.25)

    def test_nonfinite_flagged(self):
        _, q = self._qc("nonfinite")
        self.assertIn("nonfinite", q["flags"])

    # ---- 警告级：记录但不拦截 ----
    def test_lowconf_is_warning_only(self):
        _, q = self._qc("lowconf")
        self.assertIn("low_confidence", q["warnings"])
        self.assertNotIn("low_confidence", q["flags"])

    def test_tactile_dead_is_warning_only(self):
        _, q = self._qc("tactile_dead")
        self.assertIn("tactile_dead", q["warnings"])
        self.assertTrue(q["pass"])

    def test_obs_only_episode_fails_but_measures_joint_angles(self):
        """只有 obs 没有 action 的 episode 不能进训练集。"""
        _, q = self._qc("obs_only")
        self.assertIn("no_action", q["flags"])
        self.assertEqual(q["metrics"]["signal_key"], "joint_angles")

    # ---- 阈值可覆盖 ----
    def test_threshold_override(self):
        path = synth_episode.make(self.eps, "lowrate")
        self.assertIn("low_rate", qc_episode(path)["flags"])
        q = qc_episode(path, {"min_rate_hz": 5.0})
        self.assertNotIn("low_rate", q["flags"])

    # ---- 空 episode 不能崩 ----
    def test_empty_episode(self):
        d = os.path.join(self.eps, "ep_empty_left")
        os.makedirs(d)
        open(os.path.join(d, "frames.jsonl"), "w").close()
        q = qc_episode(d)
        self.assertFalse(q["pass"])
        self.assertIn("empty", q["flags"])
        self.assertIn("no_meta", q["warnings"])

    # ---- 主机时钟兜底 ----
    def test_host_clock_fallback(self):
        path = synth_episode.make(self.eps, "none")
        fp = os.path.join(path, "frames.jsonl")
        rows = load_frames(path)
        with open(fp, "w") as f:
            for r in rows:
                r.pop("t_dev_us", None)
                f.write(json.dumps(r) + "\n")
        q = qc_episode(path)
        self.assertEqual(q["metrics"]["time_source"], "host")
        self.assertIn("host_clock_only", q["warnings"])
        self.assertTrue(q["pass"])

    # ---- 分流 ----
    def test_route_link_separates_clean_and_rejected(self):
        good = synth_episode.make(self.eps, "none")
        bad = synth_episode.make(self.eps, "static")
        clean = os.path.join(self.tmp, "clean")
        rej = os.path.join(self.tmp, "rejected")
        for p in (good, bad):
            q = qc_episode(p)
            route(p, q["pass"], clean, rej, "link")
        self.assertEqual(os.listdir(clean), [os.path.basename(good)])
        self.assertEqual(os.listdir(rej), [os.path.basename(bad)])
        # 软链要能真的读到帧
        linked = os.path.join(clean, os.path.basename(good))
        self.assertEqual(len(load_frames(linked)), 200)

    def test_route_is_idempotent(self):
        good = synth_episode.make(self.eps, "none")
        clean = os.path.join(self.tmp, "clean")
        rej = os.path.join(self.tmp, "rejected")
        for _ in range(2):
            route(good, True, clean, rej, "link")
        self.assertEqual(len(os.listdir(clean)), 1)


class MetricsTest(unittest.TestCase):
    def test_metrics_on_empty_list(self):
        self.assertEqual(compute_metrics([])["n_frames"], 0)

    def test_metrics_single_frame(self):
        m = compute_metrics([{"i": 0, "t_host": 1.0, "seq": 1, "action": [0.0] * 20}])
        self.assertEqual(m["n_frames"], 1)
        self.assertEqual(m["duration_s"], 0.0)
        self.assertIsNone(m["rate_hz_median"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
