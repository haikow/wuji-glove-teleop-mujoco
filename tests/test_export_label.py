#!/usr/bin/env python3
"""导出筛选/分组守卫 + 标注工具的单测。

前半段（collect/分组/标注）只用标准库；最后一个用例真的跑一次 LeRobot 导出并读回，
需要 venv312（lerobot + torch），没装就自动跳过。

    ./venv312/bin/python tests/test_export_label.py
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import synth_episode  # noqa: E402
from tools.episode_format import load_meta, write_meta  # noqa: E402
from tools.export_dataset import (  # noqa: E402
    _obs_names, _obs_vector, collect, group_key, model_identity, parse_obs_mode,
)
from tools.label_episode import apply_label  # noqa: E402
from tools.qc_episode import qc_episode  # noqa: E402

try:
    import lerobot  # noqa: F401
    HAVE_LEROBOT = True
except Exception:
    HAVE_LEROBOT = False


class ExportSelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="export_test_")
        self.eps = os.path.join(self.tmp, "episodes")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make(self, defect="none", task="t", eid=None, **meta_over):
        p = synth_episode.make(self.eps, defect, task=task, episode_id=eid)
        qc_episode(p)
        if meta_over:
            m = load_meta(p)
            m.update(meta_over)
            write_meta(p, m)
        return p

    def test_obs_vector_dims(self):
        p = self._make(eid="ep_a")
        from tools.episode_format import load_frames
        f = load_frames(p)[0]
        self.assertEqual(len(_obs_vector(f, "skeleton")), 63)
        self.assertEqual(len(_obs_vector(f, "joints")), 20)
        self.assertEqual(len(_obs_vector(f, "both")), 83)

    def test_parse_obs_mode(self):
        self.assertEqual(parse_obs_mode("both"), ("skeleton", "joints"))
        self.assertEqual(parse_obs_mode("skeleton,hand"), ("skeleton", "hand"))
        with self.assertRaises(SystemExit):
            parse_obs_mode("skeleton,bogus")

    def test_hand_component_requires_hand_state(self):
        """没接真机的 episode 用 --obs hand 时该丢帧，而不是产生残缺向量。"""
        p = self._make(eid="ep_nohand")
        from tools.episode_format import load_frames
        f = load_frames(p)[0]
        self.assertIsNone(_obs_vector(f, "skeleton,hand"))
        f["hand_state"] = [0.1] * 20
        self.assertEqual(len(_obs_vector(f, "skeleton,hand")), 83)
        self.assertEqual(len(_obs_vector(f, "skeleton,joints,hand")), 103)

    def test_partial_hand_state_rejected(self):
        """joint_states 缺关节（某些 nid 离线）时不能用 None 混进训练向量。"""
        p = self._make(eid="ep_partial")
        from tools.episode_format import load_frames
        f = load_frames(p)[0]
        f["hand_state"] = [0.1] * 19 + [None]
        self.assertIsNone(_obs_vector(f, "hand"))

    def test_obs_names_match_dims(self):
        names = _obs_names("skeleton,joints,hand", 25, ["j%d" % i for i in range(20)])
        self.assertEqual(len(names), 63 + 25 + 20)
        self.assertEqual(names[0], "sk0_x")
        self.assertEqual(names[-1], "hand_j19")

    def test_qc_fail_is_excluded(self):
        self._make("none", eid="ep_good")
        self._make("static", eid="ep_bad")     # QC 判 near_static
        kept, skipped = collect(self.eps)
        self.assertEqual([os.path.basename(e) for e, _, _ in kept], ["ep_good"])
        self.assertIn(("qc_fail"), [w for _, w in skipped])

    def test_no_qc_filter_lets_everything_through(self):
        self._make("none", eid="ep_good")
        self._make("static", eid="ep_bad")
        kept, _ = collect(self.eps, qc_filter=False)
        self.assertEqual(len(kept), 2)

    def test_labeled_failure_excluded_by_default(self):
        p = self._make("none", eid="ep_fail")
        apply_label(p, success=False)
        kept, skipped = collect(self.eps)
        self.assertEqual(kept, [])
        self.assertIn("labeled_failure", [w for _, w in skipped])
        kept2, _ = collect(self.eps, include_failures=True)
        self.assertEqual(len(kept2), 1)

    def test_task_filter(self):
        self._make("none", task="pick", eid="ep_pick")
        self._make("none", task="place", eid="ep_place")
        kept, _ = collect(self.eps, task="pick")
        self.assertEqual([m["task"] for _, m, _ in kept], ["pick"])

    def test_obs_only_episode_has_no_aligned_frames(self):
        """没有 action 的 episode 不能进数据集，即使放宽 QC。"""
        self._make("obs_only", eid="ep_obs")
        kept, skipped = collect(self.eps, qc_filter=False)
        self.assertEqual(kept, [])
        self.assertIn("no_aligned_frames", [w for _, w in skipped])

    # ---- 溯源分组守卫 ----
    def test_model_identity_prefers_urdf_source_path(self):
        """urdf_source_path 才是硬证据；hand_model_path 只作老数据兼容。"""
        self.assertEqual(model_identity({"urdf_source_path": "/a.urdf",
                                         "hand_model_path": "/b.urdf"}), "/a.urdf")
        self.assertEqual(model_identity({"hand_model_path": "/b.urdf"}), "/b.urdf")
        self.assertEqual(model_identity({"urdf_source": "builtin_default",
                                         "hand_model": "wuji_hand"}),
                         "builtin_default/wuji_hand")

    def test_builtin_and_calibrated_are_different_groups(self):
        """内置 URDF 录的和标定后录的不能混进一个数据集。"""
        self._make("none", eid="ep_builtin", urdf_source="builtin_default",
                   urdf_source_path=None, hand_model_path=None)
        self._make("none", eid="ep_calib", urdf_source="calibration_file",
                   urdf_source_path="/users/u_x/models/c.urdf", hand_model_path=None)
        kept, _ = collect(self.eps)
        groups = {group_key(m, len(f[0]["action"]), len(_obs_vector(f[0], "both")))
                  for _, m, f in kept}
        self.assertEqual(len(groups), 2)

    def test_different_hand_model_forms_separate_groups(self):
        """不同手 URDF 的 episode 必须分成不同组，不能悄悄混进一个数据集。"""
        self._make("none", eid="ep_builtin", hand_model_path="/models/builtin.urdf")
        self._make("none", eid="ep_calib", hand_model_path="/models/user_a.urdf")
        kept, _ = collect(self.eps)
        groups = {group_key(m, len(f[0]["action"]), len(_obs_vector(f[0], "both")))
                  for _, m, f in kept}
        self.assertEqual(len(groups), 2)

    def test_same_provenance_forms_one_group(self):
        self._make("none", eid="ep_1", hand_model_path="/models/x.urdf")
        self._make("none", eid="ep_2", hand_model_path="/models/x.urdf")
        kept, _ = collect(self.eps)
        groups = {group_key(m, len(f[0]["action"]), len(_obs_vector(f[0], "both")))
                  for _, m, f in kept}
        self.assertEqual(len(groups), 1)


class LabelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="label_test_")
        self.p = synth_episode.make(os.path.join(self.tmp, "eps"), "none")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_set_and_clear_success(self):
        apply_label(self.p, success=True)
        self.assertIs(load_meta(self.p)["success"], True)
        apply_label(self.p, success=False)
        self.assertIs(load_meta(self.p)["success"], False)
        apply_label(self.p, success=None)
        self.assertIsNone(load_meta(self.p)["success"])

    def test_untouched_fields_survive(self):
        """只改 notes 不能把 success 冲掉（... 哨兵语义）。"""
        apply_label(self.p, success=True)
        apply_label(self.p, notes="hello")
        m = load_meta(self.p)
        self.assertIs(m["success"], True)
        self.assertEqual(m["notes"], "hello")

    def test_provenance_survives_labeling(self):
        before = load_meta(self.p)
        apply_label(self.p, success=True, intervention=True)
        after = load_meta(self.p)
        for k in ("glove_sn", "hand_model_path", "sdk_version", "action_space"):
            self.assertEqual(after.get(k), before.get(k))
        self.assertTrue(after["intervention"])


@unittest.skipUnless(HAVE_LEROBOT, "需要 lerobot")
class SkipEmbedImagesTest(unittest.TestCase):
    """跳过 embed_images 的安全边界：只有真没有媒体列时才跳。

    这个优化把导出从 2887 提到 12782 帧/s（真机 19 条数据），但如果在有相机的
    数据集上误跳，图像字节不会被嵌进 parquet —— 属于静默丢数据，必须挡住。
    """

    def setUp(self):
        import datasets as hfds
        import lerobot.datasets.dataset_writer as dw
        self.hfds, self.dw = hfds, dw
        self.calls = []
        self._orig = dw.embed_images
        dw.embed_images = lambda d: (self.calls.append(d) or d)

    def tearDown(self):
        self.dw.embed_images = self._orig

    class _FakeDS:
        def __init__(self, features):
            self.features = features

    def test_skips_when_no_media_columns(self):
        from tools.export_dataset import skip_embed_images_when_no_media
        ds = self._FakeDS({"observation.state": self.hfds.Value("float32"),
                           "action": self.hfds.Value("float32")})
        with skip_embed_images_when_no_media(True) as active:
            self.assertTrue(active)
            self.dw.embed_images(ds)
        self.assertEqual(self.calls, [])          # 原实现没被调用

    def test_does_not_skip_when_image_column_present(self):
        from tools.export_dataset import skip_embed_images_when_no_media
        ds = self._FakeDS({"observation.state": self.hfds.Value("float32"),
                           "observation.images.cam": self.hfds.Image()})
        with skip_embed_images_when_no_media(True) as active:
            self.assertTrue(active)
            self.dw.embed_images(ds)
        self.assertEqual(len(self.calls), 1)      # 有图像 → 走官方实现

    def test_disabled_restores_original(self):
        from tools.export_dataset import skip_embed_images_when_no_media
        with skip_embed_images_when_no_media(False) as active:
            self.assertFalse(active)
        self.assertIsNot(self.dw.embed_images, self._orig)   # setUp 的桩仍在
        # 退出上下文后不能把别人的补丁擦掉
        ds = self._FakeDS({"a": self.hfds.Value("float32")})
        self.dw.embed_images(ds)
        self.assertEqual(len(self.calls), 1)


@unittest.skipUnless(HAVE_LEROBOT, "需要 lerobot")
class ExportRoundTripTest(unittest.TestCase):
    """真的导出一次并用 LeRobotDataset 读回 —— 证明格式确实能被训练栈消费。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="export_rt_")
        self.eps = os.path.join(self.tmp, "episodes")
        for i in range(2):
            p = synth_episode.make(self.eps, "none", n=40,
                                   episode_id="ep_rt_%d" % i)
            qc_episode(p)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_export_and_reload(self):
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = os.path.join(self.tmp, "ds")
        r = subprocess.run(
            [sys.executable, os.path.join(root, "tools", "export_dataset.py"),
             "--input", self.eps, "--out", out, "--repo-id", "local/test_rt",
             "--no-qc-filter", "--overwrite"],
            capture_output=True, text=True, cwd=root)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(repo_id="local/test_rt", root=out)
        self.assertEqual(ds.meta.total_episodes, 2)
        self.assertEqual(ds.meta.total_frames, 80)
        s = ds[0]
        self.assertEqual(tuple(s["observation.state"].shape), (83,))
        self.assertEqual(tuple(s["action"].shape), (20,))
        # v3 布局的关键文件
        for rel in ("meta/info.json", "meta/stats.json", "meta/tasks.parquet"):
            self.assertTrue(os.path.isfile(os.path.join(out, rel)), rel)
        # 溯源清单
        with open(os.path.join(out, "wuji_provenance.json")) as f:
            self.assertEqual(len(json.load(f)["episodes"]), 2)

    def test_mixed_provenance_refused(self):
        """不同手模型混导必须报错退出（返回码 2），而不是默默合并。"""
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        m = load_meta(os.path.join(self.eps, "ep_rt_0"))
        m["hand_model_path"] = "/models/other.urdf"
        write_meta(os.path.join(self.eps, "ep_rt_0"), m)
        r = subprocess.run(
            [sys.executable, os.path.join(root, "tools", "export_dataset.py"),
             "--input", self.eps, "--out", os.path.join(self.tmp, "ds2"),
             "--repo-id", "local/test_rt2", "--no-qc-filter"],
            capture_output=True, text=True, cwd=root)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("拒绝混合导出", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
