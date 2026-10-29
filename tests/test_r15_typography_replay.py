"""Synthetic-only upstream parity, fold isolation and false-evidence rejection."""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from scipy import special
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import r15_typography_replay as r


def original_block():
    nb = json.loads((ROOT / 'notebooks/reference_public660_v32.ipynb').read_bytes())
    source = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'][2]
    protocol = json.loads(r.PROTOCOL.read_text(encoding='utf-8'))
    assert hashlib.sha256(source.encode()).hexdigest() == protocol['source_infer_cell_sha256']
    tree = ast.parse(source.split('\n', 1)[1])
    tail = next(n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'not a.sim')
    def assigned(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i, n in enumerate(tail.body) if assigned(n, '_marks'))
    end = next(i for i, n in enumerate(tail.body) if assigned(n, '_safe_logits'))
    # Execute only the actual arithmetic block: no filesystem/model/subprocess nodes.
    nodes = tail.body[start:end + 1]
    assert all(isinstance(n, (ast.Assign, ast.AugAssign)) for n in nodes)
    return compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-V32-typography>', 'exec')


def upstream(texts, truth, valid_texts, raw):
    env = {'np': np, 're': re, '_v38sp': special, 'LABELS': list(r.LABELS), 'y': truth,
           'train': SimpleNamespace(text=SimpleNamespace(values=texts)),
           'test': SimpleNamespace(text=SimpleNamespace(values=valid_texts)),
           '_v38_before_hard': raw.copy()}
    exec(original_block(), env)
    return env


def fixture():
    # Every identity/text/label assignment is artificial; no official rows are read.
    groups = np.repeat([f'synthetic-p{i:03}' for i in range(120)], 18)
    truth = np.tile(np.arange(18), 120)
    keep = np.arange(len(truth)) != 0
    groups, truth = groups[keep], truth[keep]
    folds = np.repeat(np.arange(120) % 5, 18)[keep]
    texts = np.asarray([(('However, — “sample”\n\n' if i % 3 == 0 else 'plain sample ') *
                        (3 + i % 19)) + ('### ``` > ' if y % 4 == 0 else 'Overall, done.')
                        for i, y in enumerate(truth)])
    n = len(truth)
    data = {'row_index': list(range(n)), 'ids': [f'synthetic-row-{i}' for i in range(n)],
            'groups': groups.tolist(), 'truth': truth.tolist(), 'labels': list(r.LABELS),
            'texts': texts.tolist(), 'fold_id': folds.tolist()}
    protocol = json.loads(r.PROTOCOL.read_text(encoding='utf-8'))
    protocol['row_identity_sha256'] = r.canonical_hash({k: data[k] for k in ('row_index', 'ids', 'groups', 'truth', 'labels')})
    protocol['text_sha256'] = r.canonical_hash(data['texts'])
    protocol['fold_sha256'] = r.canonical_hash(data['fold_id'])
    raw = np.random.default_rng(15).normal(0, .15, (n, 18))
    raw[::31, 0] = 20
    manifest = {k: protocol[k] for k in ('r14_candidate_sha256', 'row_identity_sha256', 'fold_sha256', 'text_sha256', 'replay_stage')}
    manifest.update(schema='ntl-full-ensemble-oof-v1', labels=list(r.LABELS),
                    protected_holdout_accesses=0, transductive_batch_scope='one_outer_validation_fold',
                    files={name: r.canonical_hash(['synthetic', name]) for name in r.BUNDLE_FILES}, folds=[])
    reference = np.empty_like(raw)
    for f in range(5):
        va = folds == f
        train_hash = r.canonical_hash(sorted(set(groups[~va])))
        item = {'fold': f, 'train_groups_sha256': train_hash,
                'valid_groups_sha256': r.canonical_hash(sorted(set(groups[va]))),
                'valid_ids_sha256': r.canonical_hash(np.asarray(data['ids'])[va].tolist()),
                'task_supervised_scope': 'outer_train_only', 'prior_scope': 'outer_train_only',
                'all_dependencies_enumerated': True,
                'members': {role: {'status': 'included', 'official_fit_groups_sha256': train_hash,
                                  'supervision_scope': 'outer_train_only',
                                  'asset_sha256': [r.canonical_hash(['synthetic-asset', role, f])],
                                  'provenance_sha256': r.canonical_hash(['synthetic-provenance', role, f])}
                            for role in r.REQUIRED_ROLES},
                'atlas_coverage': {'selected': int(va.sum()), 'scored': int(va.sum()) - 1,
                                   'validation_rows': int(va.sum()),
                                   'scored_ids_sha256': r.canonical_hash(np.asarray(data['ids'])[va][:-1].tolist())}}
        for key in ('producer_source_sha256', 'producer_config_sha256', 'external_overlap_audit_sha256', 'execution_receipt_sha256'):
            item[key] = r.canonical_hash(['synthetic', f, key])
        manifest['folds'].append(item)
        reference[va] = upstream(texts[~va], truth[~va], texts[va], raw[va])['_safe_logits']
    return data, raw, reference, manifest, protocol


class R15Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = fixture()

    def test_exact_v32_full_strength_parity(self):
        data, raw, reference, manifest, protocol = self.bundle
        report, base, candidate = r.replay(data, raw, reference, manifest, protocol)
        np.testing.assert_allclose(base, reference, atol=1e-13, rtol=0)
        np.testing.assert_allclose(candidate - raw, .5 * (reference - raw), atol=1e-13, rtol=0)
        self.assertEqual(report['typography_statistic_fits'], 5)
        self.assertIsNone(report['public_score'])
        self.assertIsNone(report['private_score'])
        self.assertAlmostEqual(report['baseline']['balanced_accuracy'],
                               balanced_accuracy_score(data['truth'], reference.argmax(1)), places=14)
        self.assertEqual(len(report['baseline']['per_class_recall']), 18)

    def test_markers_gates_bounds_and_no_input_mutation(self):
        truth = np.tile(np.arange(18), 20)
        texts = np.asarray([('However, — “x”\n\n' if y % 2 else 'plain x ') * 30 for y in truth])
        valid = np.asarray(['', 'short', 'However, — “x” ' * 100, '### ``` > \n1.'])
        raw = np.zeros((4, 18)); raw[0, 0] = 20
        before = raw.copy()
        reference = upstream(texts, truth, valid, raw)
        self.assertEqual(tuple(reference['_marks']), r.MARKERS)
        self.assertEqual(len(r.MARKERS), 37)
        state = r.fit_typography(texts, truth)
        base = r.apply_typography(state, raw, valid)
        half = r.apply_typography(state, raw, valid, .5)
        np.testing.assert_allclose(base, reference['_safe_logits'], atol=1e-13, rtol=0)
        np.testing.assert_array_equal(raw, before)
        np.testing.assert_array_equal(base[0], raw[0])
        self.assertLessEqual(np.max(np.abs(half - raw)), .15)

    def test_no_class_signal_has_no_effect(self):
        truth = np.tile(np.arange(18), 10)
        state = r.fit_typography(['identical text'] * len(truth), truth)
        raw = np.random.default_rng(1).normal(size=(3, 18))
        result = r.apply_typography(state, raw, ['—', 'plain', '###'], .5)
        np.testing.assert_allclose(result, raw, atol=1e-13)

    def test_bad_numeric_input_rejected(self):
        state = r.fit_typography(['x'] * 18, np.arange(18))
        for raw in (np.zeros((1, 17)), np.full((1, 18), np.nan), np.ones((1, 18), int)):
            with self.assertRaises(ValueError):
                r.apply_typography(state, raw, ['x'])
        with self.assertRaises(ValueError):
            r.apply_typography(state, np.zeros((1, 18)), ['x'], .25)
        with self.assertRaises(ValueError):
            r.fit_typography(['x'] * 17, np.arange(17))

    def test_training_receipt_rejections_before_fit(self):
        data, raw, reference, manifest, protocol = self.bundle
        changes = [lambda m: m.update(schema='component-oof'),
                   lambda m: m.update(replay_stage='save_lp_divided_by_wtot'),
                   lambda m: m.update(protected_holdout_accesses=1),
                   lambda m: m.update(protected_holdout_accesses=False),
                   lambda m: m.update(transductive_batch_scope='all_oof_rows'),
                   lambda m: m['folds'][0].update(task_supervised_scope='full_fit'),
                   lambda m: m['folds'][0].update(prior_scope='full_official_train'),
                   lambda m: m['folds'][0]['members'].pop('comp.ultrabert'),
                   lambda m: m['folds'][0]['members']['main.q14v5'].update(official_fit_groups_sha256='a'*64),
                   lambda m: m['folds'][0]['members']['main.q14v5'].update(status='upstream_optional_absent'),
                   lambda m: m['folds'][0]['atlas_coverage'].update(scored=99999)]
        for change in changes:
            altered = copy.deepcopy(manifest); change(altered)
            with mock.patch.object(r, 'fit_typography', side_effect=AssertionError('must reject before fitting')):
                with self.assertRaises(ValueError):
                    r.replay(data, raw, reference, altered, protocol)

    def test_identity_and_prompt_split_rejected_before_fit(self):
        data, raw, reference, manifest, protocol = self.bundle
        for field in ('ids', 'texts', 'fold_id', 'truth'):
            altered = copy.deepcopy(data); altered[field] = altered[field][::-1]
            with mock.patch.object(r, 'fit_typography', side_effect=AssertionError('must reject before fitting')):
                with self.assertRaises(ValueError):
                    r.replay(altered, raw, reference, manifest, protocol)
        altered = copy.deepcopy(data)
        altered['fold_id'][0] = (altered['fold_id'][0] + 1) % 5
        p = copy.deepcopy(protocol); p['fold_sha256'] = r.canonical_hash(altered['fold_id'])
        with self.assertRaisesRegex(ValueError, 'Prompt leakage'):
            r.validate_development(altered, p)

    def test_fold_receipt_binding_before_fit(self):
        data, raw, reference, manifest, protocol = self.bundle
        altered = copy.deepcopy(manifest)
        altered['folds'][0]['valid_ids_sha256'] = 'a' * 64
        with mock.patch.object(r, 'fit_typography', side_effect=AssertionError('must reject before fitting')):
            with self.assertRaisesRegex(ValueError, 'Fold provenance'):
                r.replay(data, raw, reference, altered, protocol)

    def test_rescaled_logits_fail_independent_parity_before_scoring(self):
        data, raw, reference, manifest, protocol = self.bundle
        with mock.patch.object(r, 'compare_predictions', side_effect=AssertionError('must not score')):
            with self.assertRaisesRegex(ValueError, 'parity failed'):
                r.replay(data, raw / 12.45, reference, manifest, protocol)

    def test_outer_validation_labels_do_not_enter_typography_fit(self):
        data, raw, reference, manifest, protocol = self.bundle
        actual = r.fit_typography
        seen = []
        def checked_fit(texts, truth):
            f = len(seen)
            tr = np.asarray(data['fold_id']) != f
            np.testing.assert_array_equal(texts, np.asarray(data['texts'])[tr])
            np.testing.assert_array_equal(truth, np.asarray(data['truth'])[tr])
            seen.append(f)
            return actual(texts, truth)
        with mock.patch.object(r, 'fit_typography', side_effect=checked_fit):
            r.replay(data, raw, reference, manifest, protocol)
        self.assertEqual(seen, list(range(5)))

    def test_prompt_bootstrap_matches_independent_group_draws(self):
        truth = np.tile(np.arange(18), 120)
        groups = np.repeat(np.arange(120), 18)
        folds = groups % 5
        base = (truth + 1) % 18
        candidate = base.copy(); candidate[groups < 60] = truth[groups < 60]
        protocol = self.bundle[-1]
        result = r.compare_predictions(base, candidate, truth, groups, folds, protocol)
        draws = np.random.default_rng(20261029).multinomial(120, np.full(120, 1/120), size=2000)
        expected = np.quantile(draws[:, :60].sum(1) / 120, [.025, .975])
        np.testing.assert_allclose(result['paired_prompt_gain_95ci'], expected, atol=1e-14)
        self.assertAlmostEqual(result['ba_gain'], .5)
        self.assertTrue(result['gate_passed'])
        no_gain = r.compare_predictions(base, base, truth, groups, folds, protocol)
        self.assertFalse(no_gain['gate_passed'])
        self.assertEqual(no_gain['paired_prompt_gain_95ci'], [0, 0])

    def test_optional_upstream_policy_is_explicit(self):
        manifest = copy.deepcopy(self.bundle[3]); protocol = self.bundle[-1]
        manifest['folds'][0]['members']['comp.tokattn_q35']['status'] = 'upstream_optional_absent'
        r.validate_manifest(manifest, protocol)
        manifest['folds'][0]['members']['comp.tokattn_q35'].pop('provenance_sha256')
        with self.assertRaises(ValueError):
            r.validate_manifest(manifest, protocol)

    def test_missing_bundle_readiness_never_loads_official_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'configs').mkdir()
            (root / r.PROTOCOL.relative_to(ROOT)).write_text(r.PROTOCOL.read_text(encoding='utf-8'), encoding='utf-8')
            with mock.patch.object(r.np, 'load', side_effect=AssertionError('No arrays may be opened')):
                result = r.readiness(root)
            self.assertEqual(result['state'], 'blocked')
            self.assertEqual(len(result['blockers']), 2)
            self.assertFalse(result['official_data_loaded'])
            self.assertEqual(result['official_label_scoring'], 0)

    def test_loader_is_disabled_before_any_file_access(self):
        with mock.patch.object(Path, 'open', side_effect=AssertionError('Must not read files')):
            with self.assertRaisesRegex(ValueError, 'not admitted'):
                r.load_reviewed_bundle('nonexistent', self.bundle[-1])

    def test_loader_binds_bytes_and_review_without_pickle(self):
        data, raw, reference, manifest, protocol = self.bundle
        m, p = copy.deepcopy(manifest), copy.deepcopy(protocol)
        p['real_evaluation_enabled'] = True  # Synthetic fixture only.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            payload = {'development.json': json.dumps(data).encode()}
            for name, array in [('raw_logits.npy', raw), ('r14_logits.npy', reference)]:
                out = io.BytesIO(); np.save(out, array, allow_pickle=False); payload[name] = out.getvalue()
            for name, value in payload.items():
                (path / name).write_bytes(value)
                m['files'][name] = hashlib.sha256(value).hexdigest()
            (path / 'full_ensemble_oof_manifest.json').write_text(json.dumps(m))
            review = {'manifest_sha256': r.canonical_hash(m), 'conclusion': 'all_fold_exclusions_verified',
                      'evidence_index_sha256': r.canonical_hash('synthetic-review')}
            (path / 'provenance_review.json').write_text(json.dumps(review))
            loaded = r.load_reviewed_bundle(path, p)
            np.testing.assert_array_equal(loaded[1], raw)
            self.assertEqual(loaded[0], data)
            (path / 'raw_logits.npy').write_bytes(payload['raw_logits.npy'] + b'tamper')
            with self.assertRaisesRegex(ValueError, 'Bundle bytes differ'):
                r.load_reviewed_bundle(path, p)

    def test_oversize_npy_header_rejected_before_allocation(self):
        data, raw, reference, manifest, protocol = self.bundle
        m, p = copy.deepcopy(manifest), copy.deepcopy(protocol)
        p['real_evaluation_enabled'] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            payload = {'development.json': json.dumps(data).encode()}
            out = io.BytesIO()
            np.lib.format.write_array_header_1_0(out, {'descr': '<f8', 'fortran_order': False,
                                                      'shape': (10**12, 18)})
            payload['raw_logits.npy'] = out.getvalue()
            out = io.BytesIO(); np.save(out, reference); payload['r14_logits.npy'] = out.getvalue()
            for name, value in payload.items():
                (path / name).write_bytes(value)
                m['files'][name] = hashlib.sha256(value).hexdigest()
            (path / 'full_ensemble_oof_manifest.json').write_text(json.dumps(m))
            review = {'manifest_sha256': r.canonical_hash(m), 'conclusion': 'all_fold_exclusions_verified',
                      'evidence_index_sha256': r.canonical_hash('synthetic-review')}
            (path / 'provenance_review.json').write_text(json.dumps(review))
            with mock.patch.object(r.np, 'load', side_effect=AssertionError('Must reject before allocation')):
                with self.assertRaisesRegex(ValueError, 'BEFORE allocation'):
                    r.load_reviewed_bundle(path, p)


if __name__ == '__main__':
    unittest.main(verbosity=2)
