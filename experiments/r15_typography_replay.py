"""CPU-only R14 typography replay. No model, raw CSV or Kaggle execution.

Typography equations are adapted from Roman Tamrazov's Apache-2.0 V32
(Kaggle 363222852), preserved in notebooks/reference_public660_v32.ipynb.
The sole R15 algorithm change is multiplying the final delta by 0.5.
Manifest checks enforce consistency, not the truth of producer attestations;
training provenance needs separate review before real data can be admitted.
"""

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/r15_protocol_20261029.json'
MARKERS = (' — ', '—', '–', '’', '‘', '“', '”', '«', '»', '"', ';', ':',
           '(', ')', '[', ']', '**', '`', '…', '•', '\n', '\n\n', 'e.g.',
           'i.e.', 'However,', 'Overall,', 'In conclusion', 'Therefore,',
           'Moreover,', '\t', ' - ', ' = ', '\n- ', '\n1.', '###', '```', '> ')
LABELS = ('claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5',
          'claude-sonnet-5', 'gemini-3.1-pro-preview', 'gemini-3.6-flash',
          'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash',
          'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
          'gpt-6-astra', 'kimi-k3', 'qwen3.8-max')
# R2's frozen row hash includes this order and truth indices in this order.
# Prediction arrays retain R14's LABELS order above; never rewrite the export.
DEVELOPMENT_LABELS = ('gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
                      'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash',
                      'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-fable-5',
                      'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5',
                      'claude-sonnet-5', 'qwen3.8-max', 'glm-5.3',
                      'glm-5.3-flash', 'kimi-k3')

# These are dependency roles, not a claimed count of trained models/adapters.
# A role receipt must enumerate ALL underlying checkpoints/heads and preprocessing.
REQUIRED_ROLES = (
    'main.q14v5', 'main.q8v5', 'main.ll8v5', 'main.n14b', 'main.q25_14',
    'main.deb', 'main.gauss_q14v5', 'main.gauss_q8v5', 'main.tfidf',
    'main.distortion', 'main.svc', 'main.grouping',
    'comp.stylegram', 'comp.stylometry_lgb', 'comp.ultrabert', 'comp.debv3l',
    'comp.electl', 'comp.desk', 'comp.lora24_public', 'comp.lora24_seed7',
    'comp.lora24_claude', 'comp.lora20_q25_instyle', 'comp.tokattn_q35',
    'comp.tokattn_deb', 'comp.gauss_q3_8b_L22', 'comp.gauss_glm9b_L8',
    'comp.gauss_q3_4b_L4', 'comp.gauss_debv3l_L24', 'comp.family_gates',
    'comp.grouping', 'external_data_selection', 'atlas', 'sinkhorn_prior')
OPTIONAL_ROLES = {'comp.tokattn_q35', 'comp.tokattn_deb'}
BUNDLE_FILES = {'development.json', 'raw_logits.npy', 'r14_logits.npy'}


def canonical_hash(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def marker_matrix(texts):
    require(len(texts) > 0 and all(isinstance(t, str) for t in texts),
            'Expected nonempty string texts')
    return np.array([[float(mark in text) for mark in MARKERS] for text in texts],
                    dtype=np.float32)


@dataclass(frozen=True)
class Typography:
    rate: np.ndarray
    global_rate: np.ndarray
    feature_weight: np.ndarray


def fit_typography(texts, truth):
    """Only the caller's outer TRAIN fold may enter this function."""
    x = marker_matrix(texts)
    truth = np.asarray(truth)
    require(truth.shape == (len(x),) and truth.dtype.kind in 'iu', 'Invalid truth')
    require(set(truth.tolist()) == set(range(18)), 'Training fold must cover 18 classes')
    freq = np.bincount(truth, minlength=18).astype(np.float64)
    hits = np.stack([x[truth == k].sum(0) for k in range(18)]).astype(np.float64)
    global_rate = (x.sum(0) + 2.0) / (len(x) + 4.0)
    rate = np.clip((hits + 24.0 * global_rate[None, :]) / (freq[:, None] + 24.0), .01, .99)
    global_rate = np.clip(global_rate, .01, .99)
    signal = np.sqrt(np.mean((rate - global_rate[None, :]) ** 2, axis=0))
    denom = max(float(np.median(signal[signal > 0])) if np.any(signal > 0) else 1.0, 1e-6)
    weight = np.clip(signal / denom, .25, 2.0)
    return Typography(rate, global_rate, weight)


def apply_typography(state, raw_logits, texts, strength=1.0):
    """Preserve ORIGINAL upstream logit scale; never consume --save_lp output."""
    require(strength in (0.5, 1.0), 'R15 permits exactly the two registered strengths')
    raw = np.asarray(raw_logits)
    require(raw.shape == (len(texts), 18) and raw.dtype.kind == 'f'
            and np.isfinite(raw).all(), 'Invalid original-scale logits')
    x = marker_matrix(texts)
    fw = state.feature_weight
    evidence = (x * fw[None, :]) @ np.log(state.rate).T
    evidence += ((1.0 - x) * fw[None, :]) @ np.log1p(-state.rate).T
    evidence -= ((x * fw[None, :]) @ np.log(state.global_rate)
                 + ((1.0 - x) * fw[None, :]) @ np.log1p(-state.global_rate))[:, None]
    evidence -= evidence.mean(1, keepdims=True)
    evidence = np.clip(1.5 * evidence / max(float(fw.sum()), 1e-6), -.30, .30)
    ordered = np.sort(softmax(raw, axis=1), axis=1)
    confidence_gate = np.clip((.60 - (ordered[:, -1] - ordered[:, -2])) / .60, 0, 1)
    words = np.array([len(re.findall(r'\b\w+\b', t)) for t in texts], dtype=np.float32)
    length_gate = np.clip(np.sqrt(np.maximum(words, 1.0) / 80.0), .40, 1.0)
    coverage_gate = np.clip(np.sqrt((x.sum(1) + 1.0) / 4.0), .45, 1.0)
    gate = confidence_gate * length_gate * coverage_gate
    return raw + strength * gate[:, None] * evidence


def validate_manifest(manifest, protocol):
    """Metadata-only contract. No prediction, label/text or checkpoint reads."""
    require(manifest.get('schema') == 'ntl-full-ensemble-oof-v1', 'Not a full-ensemble OOF bundle')
    for key in ('r14_candidate_sha256', 'row_identity_sha256', 'fold_sha256', 'text_sha256'):
        require(manifest.get(key) == protocol[key], f'Mismatch: {key}')
    require(manifest.get('replay_stage') == protocol['replay_stage'], 'Wrong replay stage/scale')
    require(manifest.get('labels') == list(LABELS), 'Class order mismatch')
    accesses = manifest.get('protected_holdout_accesses')
    require(type(accesses) is int and accesses == 0, 'Protected holdout is closed; integer zero required')
    require(manifest.get('transductive_batch_scope') == 'one_outer_validation_fold',
            'Cross-fold transductive mixing is forbidden')
    files = manifest.get('files', {})
    require(set(files) == BUNDLE_FILES and all(is_digest(v) for v in files.values()),
            'Missing/hashless fixed bundle files')
    folds = manifest.get('folds', [])
    require([f.get('fold') for f in folds] == list(range(protocol['folds'])), 'Incomplete/duplicate folds')
    for fold in folds:
        require(all(is_digest(fold.get(k)) for k in ('train_groups_sha256', 'valid_groups_sha256',
                'valid_ids_sha256', 'producer_source_sha256', 'producer_config_sha256',
                'external_overlap_audit_sha256', 'execution_receipt_sha256')), 'Missing fold provenance')
        require(fold.get('task_supervised_scope') == 'outer_train_only', 'Full-fit/unknown training scope')
        require(fold.get('prior_scope') == 'outer_train_only', 'Full-train prior is forbidden')
        require(fold.get('all_dependencies_enumerated') is True, 'Incomplete upstream dependency inventory')
        members = fold.get('members', {})
        require(set(members) == set(REQUIRED_ROLES), 'Incomplete configured member inventory')
        for role, member in members.items():
            require(is_digest(member.get('provenance_sha256')), f'Missing provenance: {role}')
            status = member.get('status')
            if status == 'upstream_optional_absent':
                require(role in OPTIONAL_ROLES, f'Unexpected skipped member: {role}')
            else:
                require(status == 'included', f'Invalid member status: {role}')
                require(member.get('official_fit_groups_sha256') in
                        (fold['train_groups_sha256'], canonical_hash([])), f'Leaking member: {role}')
                require(member.get('supervision_scope') in ('outer_train_only', 'no_task_supervision'),
                        f'Unknown supervision: {role}')
                expected = (canonical_hash([]) if member['supervision_scope'] == 'no_task_supervision'
                            else fold['train_groups_sha256'])
                require(member['official_fit_groups_sha256'] == expected, f'Scope inconsistency: {role}')
                assets = member.get('asset_sha256', [])
                require(bool(assets) and all(is_digest(a) for a in assets), f'Missing asset identity: {role}')
        coverage = fold.get('atlas_coverage', {})
        selected, scored, rows = (coverage.get(k) for k in ('selected', 'scored', 'validation_rows'))
        require(all(type(x) is int for x in (selected, scored, rows)) and 0 < rows
                and 0 <= scored <= selected <= rows, 'Invalid ATLAS coverage')
        require(is_digest(coverage.get('scored_ids_sha256')), 'Missing ATLAS row identity')
    return manifest


def validate_development(data, protocol):
    keys = ('row_index', 'ids', 'groups', 'truth', 'labels')
    require(canonical_hash({k: data[k] for k in keys}) == protocol['row_identity_sha256'],
            'Established 120-prompt identity changed')
    require(canonical_hash(data['fold_id']) == protocol['fold_sha256'], 'Established fold assignment changed')
    require(canonical_hash(data['texts']) == protocol['text_sha256'], 'Development text identity changed')
    n = protocol['development_rows']
    require(all(len(data[k]) == n for k in ('row_index', 'ids', 'groups', 'truth', 'texts', 'fold_id')),
            'Development row counts differ')
    require(len(set(data['ids'])) == n and len(set(data['row_index'])) == n, 'Duplicate rows')
    require(data['labels'] == list(DEVELOPMENT_LABELS), 'Historical development label order differs')
    require(len(set(data['groups'])) == protocol['development_prompts'], 'Expected exactly 120 development prompts')
    truth, folds, groups = (np.asarray(data[k]) for k in ('truth', 'fold_id', 'groups'))
    require(truth.dtype.kind in 'iu' and set(truth) == set(range(18)), 'Invalid truth labels')
    require(folds.dtype.kind in 'iu' and set(folds) == set(range(protocol['folds'])), 'Invalid folds')
    for number in range(protocol['folds']):
        tr, va = folds != number, folds == number
        require(not (set(groups[tr]) & set(groups[va])), 'Prompt leakage between folds')
        require(set(truth[va]) == set(range(18)) and set(truth[tr]) == set(range(18)),
                'Every fold must contain all classes')
    return truth, folds, groups


def truth_in_prediction_order(truth, development_labels):
    """Translate indices by class name without changing frozen development data."""
    require(list(development_labels) == list(DEVELOPMENT_LABELS),
            'Historical development label order differs')
    truth = np.asarray(truth)
    require(truth.ndim == 1 and truth.dtype.kind in 'iu'
            and np.isin(truth, np.arange(18)).all(), 'Invalid development truth')
    mapping = np.asarray([LABELS.index(label) for label in development_labels], dtype=np.int64)
    return mapping[truth]


def validate_fold_binding(manifest, data, folds, groups):
    ids = np.asarray(data['ids'])
    for item in manifest['folds']:
        va = folds == item['fold']
        expected = {'train_groups_sha256': canonical_hash(sorted(set(groups[~va].tolist()))),
                    'valid_groups_sha256': canonical_hash(sorted(set(groups[va].tolist()))),
                    'valid_ids_sha256': canonical_hash(ids[va].tolist())}
        require(all(item[k] == v for k, v in expected.items()), 'Fold provenance does not bind to rows')
        require(item['atlas_coverage']['validation_rows'] == int(va.sum()), 'ATLAS fold rows differ')


def load_reviewed_bundle(directory, protocol):
    """Bounded local loader for a FUTURE admitted run; never reads raw train.csv.

    File hashes bind the actual bytes consumed, using the same bytes for parsing
    to avoid a check/read race. No pickle is accepted. An attested review remains
    a trust dependency: its mere presence cannot prove a checkpoint's history.
    """
    require(protocol['real_evaluation_enabled'], 'Predictive execution is not admitted')
    directory = Path(directory).resolve()
    def bounded_bytes(name):
        target = (directory / name).resolve()
        require(target.parent == directory, 'Bundle file must stay in its directory')
        require(target.stat().st_size <= 32 * 1024**2, 'Bundle file exceeds 32 MiB cap')
        with target.open('rb') as stream:
            value = stream.read(32 * 1024**2 + 1)
        require(len(value) <= 32 * 1024**2, 'Bundle grew beyond 32 MiB cap')
        return value
    manifest = json.loads(bounded_bytes('full_ensemble_oof_manifest.json'))
    validate_manifest(manifest, protocol)
    review = json.loads(bounded_bytes('provenance_review.json'))
    require(review.get('manifest_sha256') == canonical_hash(manifest)
            and review.get('conclusion') == 'all_fold_exclusions_verified'
            and is_digest(review.get('evidence_index_sha256')), 'Unreviewed producer provenance')
    payload = {}
    for name, digest in manifest['files'].items():
        payload[name] = bounded_bytes(name)
        require(hashlib.sha256(payload[name]).hexdigest() == digest, f'Bundle bytes differ: {name}')
    data = json.loads(payload['development.json'])
    truth, folds, groups = validate_development(data, protocol)
    validate_fold_binding(manifest, data, folds, groups)
    arrays = []
    for name in ('raw_logits.npy', 'r14_logits.npy'):
        stream = io.BytesIO(payload[name])
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), 'Unsupported NPY version')
        header_reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                         else np.lib.format.read_array_header_2_0)
        shape, _, dtype = header_reader(stream)
        require(shape == (len(truth), 18) and dtype in (np.dtype('float32'), np.dtype('float64')),
                'Reject NPY shape/dtype BEFORE allocation')
        stream.seek(0)
        array = np.load(stream, allow_pickle=False)
        require(isinstance(array, np.ndarray) and array.shape == (len(truth), 18)
                and array.dtype in (np.dtype('float32'), np.dtype('float64'))
                and np.isfinite(array).all(), f'Invalid prediction array: {name}')
        arrays.append(array)
    return data, arrays[0], arrays[1], manifest


def metrics(pred, truth, folds):
    recall = [float(np.mean(pred[truth == k] == k)) for k in range(18)]
    per_fold = [float(np.mean([np.mean(pred[(folds == f) & (truth == k)] == k)
                               for k in range(18)])) for f in range(5)]
    return {'balanced_accuracy': float(np.mean(recall)), 'per_class_recall': dict(zip(LABELS, recall)),
            'fold_balanced_accuracy': per_fold}


def compare_predictions(baseline, candidate, truth, groups, folds, protocol):
    """Paired whole-prompt bootstrap; conditional on fixed transductive predictions."""
    baseline, candidate, truth, groups, folds = map(np.asarray, (baseline, candidate, truth, groups, folds))
    require(all(a.shape == truth.shape for a in (baseline, candidate, groups, folds))
            and truth.ndim == 1, 'Metric row shapes differ')
    require(all(a.dtype.kind in 'iu' and np.isin(a, np.arange(18)).all()
                for a in (baseline, candidate, truth)), 'Invalid metric labels')
    require(set(folds) == set(range(5)) and all(set(truth[folds == f]) == set(range(18)) for f in range(5)),
            'Metrics require five folds covering all classes')
    require(all(len(set(folds[groups == g])) == 1 for g in np.unique(groups)), 'Metric prompt crosses folds')
    base = metrics(baseline, truth, folds)
    cand = metrics(candidate, truth, folds)
    difference = (candidate == truth).astype(np.int64) - (baseline == truth).astype(np.int64)
    unique, gi = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(unique), 18), np.int64)
    hits = np.zeros_like(counts)
    np.add.at(counts, (gi, truth), 1)
    np.add.at(hits, (gi, truth), difference)
    draws = np.random.default_rng(protocol['bootstrap_seed']).multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=protocol['bootstrap_resamples'])
    denominators = draws @ counts
    require(np.all(denominators > 0), 'Bootstrap missing a class')
    ci = np.quantile(((draws @ hits) / denominators).mean(1), [.025, .975]).tolist()
    fold_gain = np.subtract(cand['fold_balanced_accuracy'], base['fold_balanced_accuracy'])
    class_gain = np.subtract(list(cand['per_class_recall'].values()), list(base['per_class_recall'].values()))
    gain = cand['balanced_accuracy'] - base['balanced_accuracy']
    g = protocol['gate']
    gates = {'gain': gain >= g['min_gain'], 'positive_ci': ci[0] > 0,
             'improved_folds': int((fold_gain > 1e-12).sum()) >= g['min_improved_folds'],
             'fold_loss': float(fold_gain.min()) >= -g['max_fold_loss'],
             'class_loss': float(class_gain.min()) >= -g['max_class_recall_loss']}
    return {'baseline': base, 'candidate': cand, 'ba_gain': gain,
            'fold_gains': fold_gain.tolist(), 'paired_prompt_gain_95ci': ci,
            'corrected': int((difference > 0).sum()), 'harmed': int((difference < 0).sum()),
            'changed_predictions': int((baseline != candidate).sum()),
            'gates': gates, 'gate_passed': all(gates.values()),
            'interval_scope': 'descriptive, fixed predictions, repeatedly used development prompts',
            'public_score': None, 'private_score': None}


def replay(data, raw_logits, upstream_r14_logits, manifest, protocol):
    """Library only; caller must first admit provenance and budget. Tests are synthetic.

    All fold bindings/identity checks precede any fit or scoring. Baseline parity
    precedes labels being used to calculate accuracy. No full model is invoked.
    """
    validate_manifest(manifest, protocol)
    truth, folds, groups = validate_development(data, protocol)
    validate_fold_binding(manifest, data, folds, groups)
    truth = truth_in_prediction_order(truth, data['labels'])
    raw, reference = np.asarray(raw_logits), np.asarray(upstream_r14_logits)
    require(raw.shape == reference.shape == (len(truth), 18)
            and raw.dtype.kind == reference.dtype.kind == 'f'
            and np.isfinite(raw).all() and np.isfinite(reference).all(), 'Invalid ensemble arrays')
    baseline, candidate = np.empty(raw.shape), np.empty(raw.shape)
    texts = np.asarray(data['texts'])
    for f in range(protocol['folds']):
        train, valid = folds != f, folds == f
        state = fit_typography(texts[train], truth[train])
        baseline[valid] = apply_typography(state, raw[valid], texts[valid], 1.0)
        candidate[valid] = apply_typography(state, raw[valid], texts[valid], .5)
    require(np.allclose(baseline, reference, atol=2e-6, rtol=0)
            and np.array_equal(baseline.argmax(1), reference.argmax(1)),
            'Independent R14 baseline parity failed; possible scale/fold/recipe drift')
    report = compare_predictions(baseline.argmax(1), candidate.argmax(1), truth, groups, folds, protocol)
    report['baseline_parity_max_abs'] = float(np.max(np.abs(baseline - reference)))
    report['typography_statistic_fits'] = protocol['folds']
    return report, baseline, candidate


def readiness(root=ROOT):
    """Never loads development.json or arrays. Missing evidence does not trigger fitting."""
    root = Path(root)
    protocol = json.loads((root / PROTOCOL.relative_to(ROOT)).read_text(encoding='utf-8'))
    path = root / protocol['expected_bundle']
    blockers = []
    if not path.is_file():
        blockers.append('Missing reviewed full-ensemble OOF manifest: ' + str(path))
    else:
        try:
            manifest = json.loads(path.read_text(encoding='utf-8'))
            validate_manifest(manifest, protocol)
            for name in sorted(BUNDLE_FILES):
                if not (path.parent / name).is_file():
                    blockers.append('Missing bundle file: ' + name)
            review_path = path.parent / 'provenance_review.json'
            if not review_path.is_file():
                blockers.append('Missing separate producer/training-provenance review')
            else:
                review = json.loads(review_path.read_text(encoding='utf-8'))
                require(review.get('manifest_sha256') == canonical_hash(manifest)
                        and review.get('conclusion') == 'all_fold_exclusions_verified'
                        and is_digest(review.get('evidence_index_sha256')),
                        'Provenance review is unbound or incomplete')
        except (OSError, ValueError, TypeError, KeyError) as error:
            blockers.append(str(error))
    if not protocol['real_evaluation_enabled']:
        blockers.append('Predictive execution disabled pending verified bundle and bounded local admission')
    return {'state': 'blocked' if blockers else 'manifest_metadata_present_file_hashes_unchecked',
            'blockers': blockers, 'official_data_loaded': False,
            'bundle_file_hashes_verified': False,
            'training_fits': 0, 'encoder_forwards': 0, 'official_label_scoring': 0,
            'protected_holdout_accesses': 0, 'public_score': None, 'private_score': None}


def main():
    # Deliberately no --run switch until real assets and an execution claim exist.
    result = readiness()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result['state'] == 'blocked' else 0


if __name__ == '__main__':
    raise SystemExit(main())
