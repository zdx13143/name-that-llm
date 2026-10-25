"""Synthetic nested isolation and scalar calibration tests; no competition data."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import softmax

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import temperature_round6 as temp
import windows_calibrated_round6 as run


class TemperatureTests(unittest.TestCase):
    def setUp(self):
        self.config,self.base = run.configuration()

    def test_scalar_learning_and_row_shift_invariance(self):
        rng = np.random.default_rng(42)
        scores = rng.normal(size=(900,3)) * .1
        probabilities = softmax(scores*6,axis=1)
        labels = np.array([rng.choice(3,p=p) for p in probabilities])
        fit = temp.fit_temperature(scores,labels,self.config["calibration"])
        self.assertLess(fit["temperature"],1)
        self.assertLess(fit["training_nll_after"],fit["training_nll_before"])
        shifted = temp.fit_temperature(scores + rng.normal(size=(900,1)),labels,self.config["calibration"])
        self.assertAlmostEqual(fit["temperature"],shifted["temperature"],places=6)
        calibrated = temp.calibrated_probabilities(scores,fit)
        np.testing.assert_array_equal(calibrated.argmax(1),scores.argmax(1))
        np.testing.assert_allclose(calibrated.sum(1),1)
        np.testing.assert_array_equal(calibrated[::-1],temp.calibrated_probabilities(scores[::-1],fit))

    def test_invalid_inputs_and_objective_limit(self):
        for invalid in (np.array([[np.nan,0,1]]),np.zeros((0,3)),np.ones(3)):
            with self.assertRaises(ValueError):
                temp.valid_scores(invalid)
        with self.assertRaises(ValueError):
            temp.nll(np.eye(3),np.array([0,1,3]))
        with self.assertRaises(ValueError):
            temp.calibrated_probabilities(np.eye(3),{"inverse_temperature":0})
        limited = {**self.config["calibration"],"max_objective_evaluations":1}
        with self.assertRaises((ValueError,RuntimeError)):
            temp.fit_temperature(np.eye(3),np.arange(3),limited)

    def test_nested_train_validation_isolation(self):
        data = {"truth":np.tile(np.arange(3),12),"groups":np.repeat(np.arange(12),3),"labels":np.arange(3)}
        train,valid = np.arange(27),np.arange(27,36)
        folds,identity = temp.nested_folds(data,train,valid,3)
        self.assertEqual(len(folds),3)
        np.testing.assert_array_equal(np.sort(np.concatenate([va for _,va,_ in folds])),train)
        for tr,va,local in folds:
            self.assertFalse(set(data["groups"][tr]) & set(data["groups"][va]))
            self.assertFalse(set(tr) & set(valid))
            self.assertFalse(set(va) & set(valid))
            np.testing.assert_array_equal(train[local],va)
        changed = copy.deepcopy(data)
        changed["truth"][valid] = 77
        _,identity_after = temp.nested_folds(changed,train,valid,3)
        np.testing.assert_array_equal(identity,identity_after)
        with self.assertRaises(ValueError):
            temp.nested_folds(data,np.arange(28),valid,3)

    def test_gaussian_scores_match_saved_formula(self):
        x = np.arange(12,dtype=float).reshape(4,3)
        model = {"mean":np.zeros(3),"scale":np.ones(3),"coefficients":np.eye(3),"intercept":np.array([.1,.2,.3])}
        actual = temp.gaussian_scores(x,model)
        np.testing.assert_allclose(softmax(actual,axis=1),softmax(x+model["intercept"],axis=1))

    def test_total_fit_budget_prevents_41st_attempt(self):
        claim = {"attempted_fits":30,"attempted_temperature_fits":10}
        with patch.object(run.dev,"consume",side_effect=AssertionError("must not consume")):
            with self.assertRaises(RuntimeError):
                run.consume(claim,"attempted_temperature_fits",self.config)

    def test_claim_blocks_before_input_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"claim.json"
            path.write_text("{}")
            with patch.object(run,"CLAIM",path),patch.object(run,"load_inputs",side_effect=AssertionError("must not load")):
                status,value = run.readiness(self.config,self.base)
            self.assertEqual(status["state"],"blocked")
            self.assertIsNone(value)


if __name__ == "__main__":
    unittest.main()
