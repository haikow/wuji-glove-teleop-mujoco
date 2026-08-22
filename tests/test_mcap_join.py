#!/usr/bin/env python3
"""obs.mcap + action.jsonl 的 join 单测。

按真机录出来的 MCAP 结构合成夹具（jsonschema schema + json 消息编码，
topic `/<SN>/hand_skeleton`，`pose.position` 是 [x,y,z] 列表、orientation 是 dict），
验证 tools/episode_format 能把两个容器拼回统一帧结构，以及 QC 能量出 join 覆盖率。

需要 mcap 库：
    ./venv312/bin/python tests/test_mcap_join.py
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mcap.writer import Writer
    HAVE_MCAP = True
except Exception:
    HAVE_MCAP = False

from tools.episode_format import load_frames, write_meta  # noqa: E402
from tools.qc_episode import qc_episode  # noqa: E402

SN = "WGTEST0000000001"
DIM = 20


def _skeleton_msg(seq, ts_us, base):
    return {
        "header": {"seq": seq, "timestamp_us": ts_us, "frame_id": "r_hand_emf_tx"},
        "joints": [{"name": "j%d" % i,
                    "pose": {"position": [base + 0.01 * i, 0.0, 0.0],
                             "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
                    "confidence": 0.9}
                   for i in range(21)],
    }


def _angles_msg(seq, ts_us, base):
    return {
        "header": {"seq": seq, "timestamp_us": ts_us, "frame_id": "r_hand_emf_tx"},
        "fingers": [{"angles": [base + 0.001 * (5 * f + k) for k in range(5)],
                     "confidence": 0.9} for f in range(5)],
    }


NIDS = [n for n in range(1, 25) if n % 5]      # 1-4/6-9/11-14/16-19/21-24


def _action_at(i):
    """和 write_actions 里一致的合成动作（锯齿），用来造真机反馈。"""
    return 0.5 + 0.4 * ((i % 40) / 40.0)


def _joint_states_msg(seq, ts_us, value, err=0.0):
    return {
        "header": {"seq": seq, "timestamp_us": ts_us, "frame_id": "hand"},
        "num_joints": 20,
        "joints": [{"nid": n, "position": value + err, "velocity": 0.0,
                    "effort": 0.0} for n in NIDS],
    }


def write_obs_mcap(path, seqs, t0_us=1_000_000, dt_us=8333,
                   hand_lag=None, hand_err=0.0):
    with open(path, "wb") as f:
        w = Writer(f)
        w.start()
        chans = {}
        topics = ["hand_skeleton", "hand_joint_angles"]
        if hand_lag is not None:
            topics.append("joint_states")
        for topic in topics:
            sid = w.register_schema(name="/%s/%s" % (SN, topic),
                                    encoding="jsonschema", data=b"{}")
            chans[topic] = w.register_channel(topic="/%s/%s" % (SN, topic),
                                              message_encoding="json", schema_id=sid)
        for n, seq in enumerate(seqs):
            ts = t0_us + n * dt_us
            for topic, msg in (("hand_skeleton", _skeleton_msg(seq, ts, 0.01 * n)),
                               ("hand_joint_angles", _angles_msg(seq, ts, 0.01 * n))):
                w.add_message(channel_id=chans[topic], log_time=ts * 1000,
                              publish_time=ts * 1000,
                              data=json.dumps(msg).encode())
            if hand_lag is not None:
                # 真机反馈 = 指令延后 hand_lag 帧再加一个恒定偏差
                src = max(n - hand_lag, 0)
                w.add_message(
                    channel_id=chans["joint_states"], log_time=ts * 1000,
                    publish_time=ts * 1000,
                    data=json.dumps(_joint_states_msg(
                        n, ts, _action_at(src), hand_err)).encode())
        w.finish()


def write_actions(path, seqs, t0_us=1_000_000, dt_us=8333, seq_index=None):
    with open(path, "w") as f:
        for n, seq in enumerate(seqs):
            i = n if seq_index is None else seq_index.get(seq, n)
            f.write(json.dumps({
                "seq": seq, "t_dev_us": t0_us + i * dt_us,
                "t_host": 1_700_000_000.0 + i * dt_us / 1e6,
                "action": [0.5 + 0.4 * ((i % 40) / 40.0)] * DIM,
                "action_raw_max_ovr": 0.0,
            }) + "\n")


@unittest.skipUnless(HAVE_MCAP, "需要 mcap 库")
class McapJoinTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mcap_join_")
        self.ep = os.path.join(self.tmp, "ep_test_right")
        os.makedirs(self.ep)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, n=200, action_seqs=None, hand_lag=None, hand_err=0.0):
        seqs = list(range(1000, 1000 + n))
        write_obs_mcap(os.path.join(self.ep, "obs.mcap"), seqs,
                       hand_lag=hand_lag, hand_err=hand_err)
        idx = {s: i for i, s in enumerate(seqs)}
        write_actions(os.path.join(self.ep, "action.jsonl"),
                      seqs if action_seqs is None else action_seqs, seq_index=idx)
        write_meta(self.ep, {"episode_id": "ep_test_right", "task": "t",
                             "side": "right", "obs_container": "mcap"})
        return seqs

    def test_frames_joined(self):
        seqs = self._build()
        fs = load_frames(self.ep)
        self.assertEqual(len(fs), len(seqs))
        self.assertEqual([f["seq"] for f in fs], seqs)
        f = fs[0]
        self.assertEqual(len(f["skeleton"]), 21)
        self.assertEqual(len(f["skeleton"][0]), 3)
        self.assertEqual(len(f["confidence"]), 21)
        self.assertEqual(len(f["joint_angles"]), 25)   # 5 指 × 5
        self.assertEqual(len(f["action"]), DIM)
        self.assertEqual(f["i"], 0)

    def test_joint_angles_joined_on_same_seq(self):
        """ja 必须来自同一 seq 的那一帧，而不是"最近可用"的那一帧。"""
        self._build(n=50)
        for f in load_frames(self.ep):
            self.assertEqual(f["ja_seq"], f["seq"])

    def test_device_timestamp_preserved(self):
        self._build(n=10)
        fs = load_frames(self.ep)
        self.assertEqual(fs[0]["t_dev_us"], 1_000_000)
        self.assertEqual(fs[1]["t_dev_us"], 1_008_333)

    def test_full_join_passes_qc(self):
        self._build()
        q = qc_episode(self.ep)
        self.assertEqual(q["metrics"]["action_join_ratio"], 1.0)
        self.assertNotIn("low_action_join", q["flags"])

    def test_partial_join_is_flagged(self):
        """retarget 回路跳帧导致 action 覆盖不全时必须判 fail，不能悄悄导出半份数据。"""
        seqs = self._build(n=200, action_seqs=list(range(1000, 1100)))   # 只有一半
        q = qc_episode(self.ep)
        self.assertEqual(len(seqs), 200)
        self.assertAlmostEqual(q["metrics"]["action_join_ratio"], 0.5, places=3)
        self.assertIn("low_action_join", q["flags"])
        self.assertFalse(q["pass"])

    def test_orphan_actions_are_dropped(self):
        """action 里有 obs.mcap 中不存在的 seq 时直接丢弃，不能造出无 obs 的帧。"""
        self._build(n=50, action_seqs=list(range(1000, 1050)) + [99999])
        fs = load_frames(self.ep)
        self.assertEqual(len(fs), 50)
        self.assertNotIn(99999, [f["seq"] for f in fs])

    def test_mcap_without_actions_fails_no_action(self):
        seqs = list(range(1000, 1200))
        write_obs_mcap(os.path.join(self.ep, "obs.mcap"), seqs)
        write_meta(self.ep, {"episode_id": "ep_test_right", "side": "right"})
        q = qc_episode(self.ep)
        self.assertEqual(q["metrics"]["action_frames"], 0)
        self.assertIn("no_action", q["flags"])
        self.assertEqual(q["metrics"]["signal_key"], "joint_angles")


    # ---- 真机手跟踪误差 ----
    def test_hand_state_joined_by_timestamp(self):
        self._build(n=100, hand_lag=0)
        fs = load_frames(self.ep)
        self.assertTrue(all(len(f["hand_state"]) == 20 for f in fs))
        q = qc_episode(self.ep)
        self.assertEqual(q["metrics"]["hand_state_ratio"], 1.0)

    def test_tracking_lag_is_recovered(self):
        """反馈滞后 5 帧时，扫描应当在 lag=5 处取到最小误差。"""
        self._build(n=200, hand_lag=5)
        m = qc_episode(self.ep)["metrics"]
        self.assertEqual(m["track_best_lag_frames"], 5)
        self.assertLess(m["track_mae_best_rad"], 1e-6)
        # 零滞后的 MAE 明显更大 —— 说明"滞后"确实被从"误差"里分离出来了
        self.assertGreater(m["track_mae_rad"], m["track_mae_best_rad"])

    def test_poor_tracking_flagged(self):
        """扣掉滞后后仍有大偏差（手跟不动）必须判 fail。"""
        self._build(n=200, hand_lag=2, hand_err=0.5)
        q = qc_episode(self.ep)
        self.assertIn("poor_tracking", q["flags"])
        self.assertGreater(q["metrics"]["track_mae_best_rad"], 0.2)

    def test_good_tracking_passes(self):
        self._build(n=200, hand_lag=2, hand_err=0.01)
        q = qc_episode(self.ep)
        self.assertNotIn("poor_tracking", q["flags"])
        self.assertLess(q["metrics"]["track_mae_best_rad"], 0.2)

    def test_no_hand_means_no_tracking_metrics(self):
        """纯仿真 episode 不该凭空冒出跟踪误差指标。"""
        self._build(n=100)
        m = qc_episode(self.ep)["metrics"]
        self.assertEqual(m["hand_state_frames"], 0)
        self.assertIsNone(m["track_mae_best_rad"])

    def test_nid_to_flat_mapping(self):
        from tools.episode_format import nid_to_flat
        self.assertEqual([nid_to_flat(n) for n in (1, 4, 6, 9, 11, 21, 24)],
                         [0, 3, 4, 7, 8, 16, 19])


if __name__ == "__main__":
    unittest.main(verbosity=2)
