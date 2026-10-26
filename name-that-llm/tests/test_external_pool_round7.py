"""Synthetic safeguards for reconstructed-prompt exclusion and fold isolation."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from external_pool_round7 import canonical_group, fold_pool, prepare_row, remove_overlaps


class ExternalPoolTests(unittest.TestCase):
    def test_reconstructed_official_prompt_is_rejected_before_text_access(self):
        row, reason = prepare_row("prvsiyan/prvsiyans-extended-name-that-llm-dataset", {"split": "train_regen"}, ["a"])
        self.assertIsNone(row)
        self.assertEqual(reason, "excluded_prvsiyan_split:train_regen")

    def test_truncated_passage_rejected_before_text_access(self):
        row, reason = prepare_row("lxdlxdlxd/ntl-synth-generations-v1", {"label": "a", "finish_reason": "length"}, ["a"])
        self.assertIsNone(row)
        self.assertEqual(reason, "unfinished_or_missing_finish")

    def test_shared_prompt_bank_has_common_group_across_sources(self):
        self.assertEqual(canonical_group("seyitkaangunes", {"prompt_id": "g0001"}),
                         canonical_group("willrice", {"prompt_id": "g0001"}))
        self.assertEqual(canonical_group("lucasboesen", {"prompt_source": "seyitkaangunes", "prompt_id": "g0001"}),
                         canonical_group("abdullah1afify", {"prompt_id": "g0001"}))

    def test_copy_screen_excludes_whole_external_group(self):
        original = "one two three four five six seven eight nine ten eleven twelve"
        rows = [{"text": original, "group": "copied"}, {"text": "other model answer", "group": "copied"},
                {"text": "unrelated outside topic", "group": "independent"}]
        kept, info = remove_overlaps(rows, [original.upper()])
        self.assertEqual([x["group"] for x in kept], ["independent"])
        self.assertEqual(info["excluded_group_rows"], 2)

    def test_punctuation_presence_uses_only_training_rows(self):
        pool = {"texts": np.asarray(["clean a", "a\u2019s", "b\u2019s", "clean b", "clean c"]),
                "truth": np.asarray([0, 0, 1, 1, 2])}
        texts = np.asarray(["plain zero", "one\u2019s", "plain two"])
        labels = np.arange(3)
        selected, present = fold_pool(pool, texts, labels, classes=3)
        self.assertEqual(selected.tolist(), [0, 2, 3, 4])
        self.assertFalse(present[0, 2])
        self.assertTrue(present[1, 2])
        # Only explicitly adding the extra row to training may change the rule.
        selected2, _ = fold_pool(pool, np.concatenate([texts, ["zero\u2019s"]]), np.r_[labels, 0], classes=3)
        self.assertEqual(selected2.tolist(), [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
