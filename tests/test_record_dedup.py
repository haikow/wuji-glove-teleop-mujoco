#!/usr/bin/env python3
"""record_episode.record_one 的离线单测：用假订阅/假 retarget 驱动录制回路。

验证录制端最关键的三条契约（不需要手套）：
  1. 只有 header.seq 变了才算新帧 —— 轮询取最新帧不能把同一帧重复算
  2. 每条 action 都带上产生它的那一帧的 seq —— 这是和 obs.mcap join 的唯一键
  3. 预热丢弃开头不稳定的帧，且帧不够时按超时退出、不卡死

recorder=None 时 record_one 不录 MCAP，所以这些用例不需要 SDK 录制器。
需要 venv312（record_episode 会 import wuji_sdk / mujoco）：
    ./venv312/bin/python tests/test_record_dedup.py
"""
import asyncio
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    from record_episode import record_one, resolve_mjcf
    HAVE_DEPS = True
except Exception:
    HAVE_DEPS = False

from tools.episode_format import McapEpisodeWriter, load_actions  # noqa: E402


class FakeHeader:
    def __init__(self, seq, ts):
        self.seq, self.timestamp_us = seq, ts


class FakePose:
    def __init__(self, p):
        self.position = p


class FakeJoint:
    def __init__(self, p, conf=0.9):
        self.pose, self.confidence = FakePose(p), conf


class FakeSkeleton:
    def __init__(self, seq, ts, base=0.0):
        self.header = FakeHeader(seq, ts)
        self.joints = [FakeJoint([base + 0.01 * i, 0.0, 0.0]) for i in range(21)]


class FakeSub:
    """每次 _drain 只吐一帧，模拟设备按帧到达（而不是一次排空整队）。"""

    def __init__(self, frames):
        self.frames = list(frames)
        self._served = False

    def recv(self):
        if self._served:
            self._served = False
            return None
        if not self.frames:
            return None
        self._served = True
        return self.frames.pop(0)


class FakeSess:
    """假 retarget：把 21×3 关键点压成 20 维动作，便于断言 obs→action 映射。"""

    def __init__(self, gain=1.0, offset=0.0):
        self.gain, self.offset, self.n_reset = gain, offset, 0

    def reset(self):
        self.n_reset += 1

    def step(self, kp):
        return np.full(20, float(kp[1][0]) * self.gain + self.offset)


@unittest.skipUnless(HAVE_DEPS, "需要 venv312（wuji_sdk / mujoco / numpy）")
class RecordLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rec_test_")
        self.jlo = np.full(20, -1.0)
        self.jhi = np.full(20, 1.0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, frames, sess=None, seconds=0.5, warmup_frames=0):
        ep = McapEpisodeWriter(self.tmp, side="left", task="t", env={})
        sess = sess or FakeSess()
        st, summary = asyncio.run(record_one(
            ep, FakeSub(frames), sess, self.jlo, self.jhi, 20, seconds,
            warmup_frames=warmup_frames))
        ep.finalize()
        self.assertIsNone(summary)      # recorder=None → 不录 MCAP
        return ep.path, st, sess

    def test_dedup_by_seq(self):
        """同一 seq 重复到达只产生一条 action；seq 变了才产生新的。"""
        seqs = [1, 1, 1, 2, 2, 3, 3, 3]
        frames = [FakeSkeleton(s, 1000 + s * 10, base=0.1 * s) for s in seqs]
        path, st, _ = self._run(frames)
        self.assertEqual([a["seq"] for a in load_actions(path)], [1, 2, 3])
        self.assertEqual(st["actions"], 3)
        self.assertEqual(st["skipped_dup"], 5)

    def test_action_carries_join_key_and_device_time(self):
        """每条 action 都要带 seq（join 键）和 t_dev_us，否则拼不回 obs。"""
        frames = [FakeSkeleton(s, 7_000_000 + s * 10_000) for s in (1, 2, 3)]
        path, _, _ = self._run(frames)
        acts = load_actions(path)
        self.assertEqual([a["seq"] for a in acts], [1, 2, 3])
        self.assertEqual([a["t_dev_us"] for a in acts],
                         [7_010_000, 7_020_000, 7_030_000])
        for a in acts:
            self.assertEqual(len(a["action"]), 20)
            self.assertGreater(a["t_host"], 0)

    def test_action_computed_from_that_frame(self):
        """action 必须由带同一个 seq 的那一帧算出，不能用上一帧/最新帧。"""
        frames = [FakeSkeleton(s, 1000 + s, base=0.1 * s) for s in (1, 2, 3, 4)]
        path, _, _ = self._run(frames, FakeSess(gain=2.0))
        for a in load_actions(path):
            expected = (0.1 * a["seq"] + 0.01) * 2.0   # FakeSess: skeleton[1].x * 2
            self.assertAlmostEqual(a["action"][0], expected, places=5)

    def test_action_is_unclipped_but_overshoot_recorded(self):
        """action 记的是 retarget 原始解（不被本地 MJCF 限位裁剪），

        因为 retarget 已在目标手型自己的 URDF 限位内解算，而本仓库 MJCF 只有一代手；
        用二代 profile 时拿一代限位裁会裁错。超出量单独记进 action_raw_max_ovr，
        正好用来发现手型不匹配。
        """
        frames = [FakeSkeleton(1, 1000), FakeSkeleton(2, 2000)]
        path, _, _ = self._run(frames, FakeSess(gain=0.0, offset=3.0))
        for a in load_actions(path):
            self.assertTrue(all(abs(v - 3.0) < 1e-9 for v in a["action"]))
            self.assertAlmostEqual(a["action_raw_max_ovr"], 2.0, places=5)

    def test_session_reset_per_episode(self):
        _, _, sess = self._run([FakeSkeleton(1, 1000)])
        self.assertEqual(sess.n_reset, 1)

    def test_no_frames_produces_empty_episode(self):
        """手套没出帧也要留下一个可被 QC 判 empty 的目录，而不是崩掉。"""
        path, st, _ = self._run([], seconds=0.2)
        self.assertEqual(st["actions"], 0)
        self.assertEqual(load_actions(path), [])
        self.assertTrue(os.path.isfile(os.path.join(path, "meta.json")))

    def test_warmup_discards_leading_frames(self):
        """预热丢掉开头不稳定的帧（真机实测首帧后还会停顿 0.33s）。"""
        frames = [FakeSkeleton(s, 1000 + s) for s in (1, 2, 3, 4, 5)]
        path, st, _ = self._run(frames, warmup_frames=2)
        self.assertEqual(st["warmup_frames"], 2)
        self.assertEqual([a["seq"] for a in load_actions(path)], [3, 4, 5])

    def test_warmup_timeout_does_not_hang(self):
        """帧不够时预热按超时退出，不能卡死。"""
        ep = McapEpisodeWriter(self.tmp, side="left", task="t", env={})
        st, _ = asyncio.run(record_one(
            ep, FakeSub([]), FakeSess(), self.jlo, self.jhi, 20, 0.1,
            warmup_frames=100, warmup_timeout_s=0.3))
        ep.finalize()
        self.assertEqual(st["warmup_frames"], 0)
        self.assertEqual(st["actions"], 0)

    def test_meta_records_action_count(self):
        frames = [FakeSkeleton(s, 1000 + s) for s in (1, 2, 3)]
        path, _, _ = self._run(frames)
        import json
        meta = json.load(open(os.path.join(path, "meta.json")))
        self.assertEqual(meta["num_frames"], 3)
        self.assertEqual(meta["num_actions"], 3)
        self.assertEqual(meta["obs_container"], "mcap")


@unittest.skipUnless(HAVE_DEPS, "需要 venv312")
class MjcfResolveTest(unittest.TestCase):
    """二代手必须用二代 MJCF —— 一代限位下 63.7% 的帧越界、最大 0.53rad。"""

    def test_gen1_uses_vendored_description(self):
        self.assertIn("wuji_hand_description/mjcf", resolve_mjcf("wuji_hand", "right"))

    def test_override_wins(self):
        self.assertEqual(resolve_mjcf("wuji_hand", "right", "/x/y.xml"), "/x/y.xml")

    def test_gen2_prefers_description2_else_falls_back(self):
        got = resolve_mjcf("wuji_hand_2", "right")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.isfile(os.path.join(root, "wuji_hand_description2",
                                       "mjcf", "right.xml")):
            self.assertIn("wuji_hand_description2", got)
        else:
            self.assertIn("wuji_hand_description/", got)   # 回落并已打印告警


if __name__ == "__main__":
    unittest.main(verbosity=2)
