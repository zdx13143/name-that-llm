"""Synthetic CPU checks only: no competition inputs, trained checkpoints or GPU forwards."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import modernbert_finetune_training as training
import windows_modernbert_finetune_round4 as runner


class Hooks:
    def __init__(self):
        self.before_calls = []
        self.after_calls = []
    def before(self, name):
        self.before_calls.append(name)
    def after(self, name):
        self.after_calls.append(name)


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Embedding(32, 6)
        self.classifier = torch.nn.Linear(6, 3)
    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids)
        pooled = (hidden * attention_mask[..., None]).sum(1) / attention_mask.sum(1, keepdim=True)
        return SimpleNamespace(logits=self.classifier(pooled))


class Round4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads(runner.CONFIG.read_bytes())

    def test_accumulation_matches_full_effective_batch_including_short_last_batch(self):
        torch.manual_seed(4)
        model = TinyClassifier()
        reference = deepcopy(model)
        encoded = {"input_ids": torch.randint(0, 32, (11, 7)), "attention_mask": torch.ones(11, 7, dtype=torch.long)}
        truth = torch.arange(11) % 3
        method = {**self.config["training"], "micro_batch_size": 3, "gradient_accumulation": 2}
        optimizer = torch.optim.SGD([{"params": model.parameters(), "lr": .05, "initial_lr": .05}])
        reference_opt = torch.optim.SGD(reference.parameters(), lr=.05)
        order = np.random.default_rng(6).permutation(11)
        hooks = Hooks()
        info, steps = training.train_epoch(model, encoded, truth, optimizer, method, order, 0, 2, hooks, torch.device("cpu"))
        for chunks in training.update_batches(order, 3, 2):
            selected = np.concatenate(chunks)
            reference_opt.zero_grad(set_to_none=True)
            logits = reference(**{key: value[selected] for key, value in encoded.items()}).logits
            torch.nn.functional.cross_entropy(logits, truth[selected]).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            reference_opt.step()
        self.assertEqual((info["examples_seen"], info["microbatches"], steps), (11, 4, 2))
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        self.assertEqual(hooks.before_calls, hooks.after_calls)
        self.assertEqual(hooks.before_calls.count("train_forward"), 4)
        self.assertFalse(any("valid" in item for item in hooks.before_calls))

    def test_repeated_or_missing_training_rows_rejected_before_computation(self):
        model, hooks = TinyClassifier(), Hooks()
        optimizer = training.optimizer_for(model, self.config["training"])
        encoded = {"input_ids": torch.ones(3, 4, dtype=torch.long), "attention_mask": torch.ones(3, 4, dtype=torch.long)}
        with self.assertRaisesRegex(ValueError, "exactly once"):
            training.train_epoch(model, encoded, torch.zeros(3, dtype=torch.long), optimizer, self.config["training"],
                                 np.array([0, 0, 2]), 0, 1, hooks, torch.device("cpu"))
        self.assertEqual(hooks.before_calls, [])

    def test_schedule_is_bounded_and_out_of_budget_rejected(self):
        factors = [training.learning_rate_factor(i, 100, .1) for i in range(100)]
        self.assertTrue(all(0 < x <= 1 for x in factors))
        self.assertEqual(factors[:10], [i / 10 for i in range(1, 11)])
        self.assertTrue(all(a >= b for a, b in zip(factors[10:], factors[11:])))
        with self.assertRaises(ValueError):
            training.learning_rate_factor(100, 100, .1)

    def test_real_architecture_backward_and_checkpoint_roundtrip_on_tiny_synthetic_model(self):
        from transformers import ModernBertConfig, ModernBertForMaskedLM, AutoModelForSequenceClassification
        configuration = ModernBertConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                        num_attention_heads=4, max_position_embeddings=32, local_attention=8,
                                        global_attn_every_n_layers=2, pad_token_id=0, cls_token_id=1, bos_token_id=1,
                                        eos_token_id=2, sep_token_id=2, reference_compile=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "pretrained"
            ModernBertForMaskedLM(configuration).save_pretrained(original, safe_serialization=True)
            with patch.object(torch.cuda, "is_available", return_value=False):
                model, _ = runner.initial_model(original, [f"class{i}" for i in range(18)], self.config["training"], 12, torch.device("cpu"))
            original_parameter = model.model.layers[0].mlp.Wi.weight.detach().clone()
            optimizer = training.optimizer_for(model, self.config["training"])
            encoded = {"input_ids": torch.randint(3, 64, (4, 16)), "attention_mask": torch.ones(4, 16, dtype=torch.long)}
            hooks = Hooks()
            info, step = training.train_epoch(model, encoded, torch.tensor([0, 1, 2, 3]), optimizer,
                                               self.config["training"], np.arange(4), 0, 1, hooks, torch.device("cpu"), phase="probe")
            self.assertEqual(step, 1)
            self.assertTrue(np.isfinite(info["mean_training_loss"]))
            self.assertFalse(torch.equal(original_parameter, model.model.layers[0].mlp.Wi.weight))
            expected = training.predict(model, encoded, 4, hooks, torch.device("cpu"), phase="probe_eval")
            manifest = runner.save_checkpoint(model, root / "saved")
            self.assertIn("model.safetensors", manifest)
            restored = AutoModelForSequenceClassification.from_pretrained(root / "saved", local_files_only=True,
                                                                           reference_compile=False, attn_implementation="sdpa")
            actual = training.predict(restored, encoded, 4, Hooks(), torch.device("cpu"))
            np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)

    def test_counts_include_every_training_validation_and_new_preflight_operation(self):
        folds = [(np.arange(1727), np.arange(432)) for _ in range(4)] + [(np.arange(1728), np.arange(431))]
        counts = runner.expected_counts(folds, self.config)
        self.assertEqual(counts, {"training_microbatches": 8640, "validation_forward_calls": 540,
                                 "training_optimizer_steps": 1080, "logical_forward_calls": 9183,
                                 "backward_calls": 8641, "optimizer_steps": 1081})

    def test_new_claim_path_and_budget_are_fail_closed(self):
        claim = runner.fresh_claim(self.config, {"synthetic": True})
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "round4.json"
            runner.write_json(target, claim)
            hooks = runner.BudgetHooks(claim, self.config, target, lambda: None)
            hooks.before("probe_forward")
            self.assertEqual(json.loads(target.read_bytes())["attempted_forward_calls"], 1)
            self.assertEqual(json.loads(target.read_bytes())["completed_forward_calls"], 0)
            hooks.after("probe_forward")
            before = target.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                hooks.before("probe_forward")
            self.assertEqual(before, target.read_bytes())

    def test_resource_failure_does_not_mark_operation_completed(self):
        claim = runner.fresh_claim(self.config, {"synthetic": True})
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "round4.json"
            runner.write_json(target, claim)
            def exceeded():
                raise MemoryError("synthetic limit")
            hooks = runner.BudgetHooks(claim, self.config, target, exceeded)
            hooks.before("probe_forward")
            with self.assertRaises(MemoryError):
                hooks.after("probe_forward")
            saved = json.loads(target.read_bytes())
            self.assertEqual((saved["attempted_forward_calls"], saved["completed_forward_calls"]), (1, 0))

    def test_metric_gate_requires_absolute_and_paired_conditions(self):
        metrics = {"balanced_accuracy": .4, "fold_balanced_accuracy": [.4] * 5}
        comparison = {"pooled_ba_gain": .1, "fold_ba_gains": [.1] * 5,
                      "prompt_paired_gain_95ci": [.05, .15], "improved_folds": 5}
        self.assertTrue(runner.metric_gate(metrics, comparison, self.config)["metric_gate_passed"])
        comparison["prompt_paired_gain_95ci"][0] = -.001
        self.assertFalse(runner.metric_gate(metrics, comparison, self.config)["metric_gate_passed"])

    def test_existing_claim_blocks_without_loading_training_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "exists.json"
            target.write_text("{}", encoding="utf-8")
            with patch.object(runner, "CLAIM", target), patch.object(runner, "load_inputs") as loader:
                state, value = runner.readiness(self.config, {})
                self.assertEqual(state["state"], "blocked")
                self.assertIsNone(value)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
