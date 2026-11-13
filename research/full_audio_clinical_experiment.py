"""Post-hoc exploratory nested-CV rerun for full audio plus clinical inputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from cpu_model_experiment import (_checkpoint, _evaluate, _finite_scores, _fit_reported, _inner_search,
                                  _measure_latency, _select, atomic_json, configure, digest)
from src import coda_research as cr
from src import coda_v4 as v4
from src import coda_v5_features as features
from src import coda_v5_models as v5
from src import coda_v6_models as models
from src import coda_v6_xai as xai

REFERENCE = 'cpu_model_available_data_001'
BASE_IDS = ('full_beats_cb_baseline', 'v5_beats_full_cb_01', 'clinical_full_beats_cb_baseline')
REPORT = {'full': ['full_beats_cb_baseline', 'v5_beats_full_cb_01', 'full_paper_cb'],
          'clinical_full': ['clinical_full_beats_cb_baseline']}
BLEND = {'id': 'full_blend_equal', 'endpoint': 'full', 'audio': ['beats'],
         'clinical': v5.CLINICAL + ['heart_rate'], 'family': 'weighted_blend',
         'component_ids': ['full_beats_cb_baseline', 'clinical_full_beats_cb_baseline'],
         'weights': [0.5, 0.5], 'role': 'late_fusion_control'}


def frozen_candidates():
    grid = {candidate['id']: candidate for candidate in models.candidate_grid()}
    paper = dict(grid['full_beats_cb_baseline'])
    paper.update(id='full_paper_cb', audio=['paper'], pca=None, role='light_audio_alternative')
    return [dict(grid[candidate_id]) for candidate_id in BASE_IDS] + [paper]


def verify_reference(reference_run, cfg):
    reference = Path(reference_run).resolve()
    if reference.name != REFERENCE:
        raise RuntimeError(f'Reference run must be {REFERENCE}, got {reference.name}')
    provenance = json.loads((reference / 'data_provenance.json').read_text(encoding='utf-8'))
    prior = json.loads((reference / 'resolved_config.json').read_text(encoding='utf-8'))
    if prior.get('input_hashes') != cfg['input_hashes']:
        raise RuntimeError('Reference input hashes differ from current manifest, metadata, or pinned BEATs weights')
    if prior['input_hashes']['manifest'] != provenance['manifest_sha256']:
        raise RuntimeError('Reference manifest SHA does not match data provenance')
    if provenance['n_participants'] != 1040 or provenance['n_coughs'] != 9281:
        raise RuntimeError('Reference cohort count differs from frozen 1,040/9,281 cohort')
    for filename, expected in provenance['source_metadata_sha256'].items():
        if cfg['input_hashes'].get(filename) != expected:
            raise RuntimeError(f'Reference metadata hash mismatch: {filename}')
    return reference


def _inner_blend(fold, results, folder, cfg):
    left, right = (results[candidate_id]['oof'] for candidate_id in BLEND['component_ids'])
    keys = ['participant', 'y', 'inner_fold']
    if left.duplicated('participant').any() or right.duplicated('participant').any():
        raise RuntimeError('Duplicate participant in component OOF rows')
    merged = left[keys + ['raw_probability']].merge(
        right[keys + ['raw_probability']], on='participant', how='outer',
        suffixes=('_beats', '_clinical'), indicator=True, validate='one_to_one')
    if (merged['_merge'] != 'both').any():
        raise RuntimeError('Missing component OOF participant for blend')
    if ((merged.y_beats != merged.y_clinical).any() or
            (merged.inner_fold_beats != merged.inner_fold_clinical).any()):
        raise RuntimeError('Mismatched label or inner_fold in blend component OOF rows')
    frame = pd.DataFrame({'participant': merged.participant, 'y': merged.y_beats,
                          'raw_probability': (merged.raw_probability_beats + merged.raw_probability_clinical) * .5,
                          'inner_fold': merged.inner_fold_beats}).sort_values('participant').reset_index(drop=True)
    key = digest({'protocol': cfg['protocol_fingerprint'], 'fold': fold, 'candidate': BLEND,
                  'oof': frame.to_dict(orient='records')})
    checkpoint_path = Path(folder) / 'inner' / 'full_blend_equal_oof.json'
    cached = _checkpoint(checkpoint_path, key, lambda: {
        'signature': key, 'candidate': BLEND, 'oof': frame.to_dict(orient='records')})
    if cached.get('candidate') != BLEND or cached.get('oof') != frame.to_dict(orient='records'):
        raise RuntimeError('Blend OOF cache signature or scores mismatch')
    aucs, prs = [], []
    for index in range(len(fold['inner_folds'])):
        part = frame[frame.inner_fold == index]
        _, score = _finite_scores(part.y.to_numpy(), part.raw_probability.to_numpy())
        aucs.append(score['roc_auc']); prs.append(score['pr_auc'])
    result = {'candidate': BLEND, 'oof': frame, 'mean_inner_roc_auc': float(np.mean(aucs)),
              'mean_inner_pr_auc': float(np.mean(prs)), 'coverage': len(frame)}
    results[BLEND['id']] = result
    prior = pd.read_csv(Path(folder) / 'inner_search.csv')
    record = {'candidate': json.dumps(BLEND, sort_keys=True), 'mean_inner_roc_auc': result['mean_inner_roc_auc'],
              'mean_inner_pr_auc': result['mean_inner_pr_auc'], 'coverage': len(frame)}
    pd.concat([prior, pd.DataFrame([record])], ignore_index=True).to_csv(Path(folder) / 'inner_search.csv', index=False)
    return result


def _fit_blend(table, fold, result, components, cfg, folder):
    import joblib
    oof = result['oof'].sort_values('participant')
    if len(oof) < 2 or oof.y.nunique() != 2:
        raise ValueError('Insufficient blend OOF rows for calibration')
    signature = digest({'protocol': cfg['protocol_fingerprint'], 'fold': fold,
                        'candidate': BLEND, 'oof_participants': oof.participant.tolist()})
    path = Path(folder) / 'models' / BLEND['id'] / 'bundle.joblib'
    sidecar = path.with_suffix('.sha256')
    if path.exists() or sidecar.exists():
        if not path.is_file() or not sidecar.is_file() or cr.sha256(path) != sidecar.read_text().strip():
            raise RuntimeError(f'Blend bundle is incomplete or changed: {path}')
        bundle = joblib.load(path)
        if bundle.get('signature') != signature or bundle.get('candidate') != BLEND:
            raise RuntimeError(f'Blend bundle fingerprint mismatch: {path}')
        return bundle
    calibrator = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip')
    calibrator.fit(oof.raw_probability, oof.y)
    model = {'candidate': BLEND, 'components': {key: components[key]['model'] for key in BLEND['component_ids']}}
    calibration = table.loc[fold['calibration']]
    raw = models.predict_raw(model, calibration)
    calibrated = calibrator.predict(raw)
    if calibration.tb_status.nunique() != 2:
        raise ValueError('Calibration participants do not include both labels')
    thresholds = {policy: v4.threshold(calibration.tb_status.to_numpy(int), calibrated, policy,
                                      cfg['target_sensitivity']) for policy in ['youden', 'sensitivity90']}
    bundle = {'model': model, 'calibrator': calibrator, 'candidate': BLEND,
              'signature': signature, 'thresholds': thresholds,
              'proper_ids': list(fold['proper_train']), 'calibration_ids': calibration.index.tolist(),
              'calibration_raw': raw.tolist(), 'calibration_probability': calibrated.tolist()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(path)
    sidecar.write_text(cr.sha256(path), encoding='ascii')
    return bundle


def _compare_fixed_baselines(reference, metrics):
    old = pd.read_csv(reference / 'outer_fold_metrics.csv')
    old = old[(old.policy == 'youden') & old.candidate.isin(BLEND['component_ids'])]
    new = metrics[(metrics.policy == 'youden') & metrics.candidate.isin(BLEND['component_ids'])]
    for candidate_id in BLEND['component_ids']:
        a = old[old.candidate == candidate_id].sort_values('outer_fold').raw_roc_auc.to_numpy()
        b = new[new.candidate == candidate_id].sort_values('outer_fold').raw_roc_auc.to_numpy()
        if len(a) != 10 or len(b) != 10 or not np.allclose(a, b, rtol=0, atol=1e-8):
            raise RuntimeError(f'Fixed baseline outer AUC mismatch: {candidate_id}')


def run(project_root, reference_run, run_name):
    cfg = configure(project_root, run_name, check_existing=False)
    reference = verify_reference(reference_run, cfg)
    prior_splits = json.loads((reference / 'nested_splits.json').read_text(encoding='utf-8'))
    cfg['source_hashes']['full_audio_clinical_experiment.py'] = cr.sha256(__file__)
    cfg['source_hashes']['src/coda_v6_models.py'] = cr.sha256(PACKAGE / 'src/coda_v6_models.py')
    cfg['source_hashes']['cpu_model_experiment.py'] = cr.sha256(PACKAGE / 'cpu_model_experiment.py')
    cfg['frozen_candidates'] = frozen_candidates() + [BLEND]
    cfg['protocol_fingerprint'] = digest({key: value for key, value in cfg.items()
                                         if key not in ['output_dir', 'package_root']})
    output = Path(cfg['output_dir'])
    resolved = output / 'resolved_config.json'
    if resolved.exists():
        prior = json.loads(resolved.read_text(encoding='utf-8'))
        if prior.get('protocol_fingerprint') != cfg['protocol_fingerprint']:
            raise RuntimeError('Augmented run fingerprint differs; existing run remains untouched')
    cr.seed_everything(cfg['seed'])
    import torch
    torch.set_num_threads(cfg['cpu_threads'])
    with threadpool_limits(limits=cfg['cpu_threads']):
        participants, manifest = v4.audit_cohort(cfg)
        table = features.prepare_features(participants, manifest, cfg, include_beats=True)
        splits = v4.nested_splits(participants, cfg)
        if splits != prior_splits:
            raise RuntimeError('Nested participant splits differ from reference; stopping before model fitting')
        if len(table) != 1040 or len(manifest) != 9281:
            raise RuntimeError('Cohort or audio manifest size differs from reference')
        all_test = [participant for fold in splits['folds'] for participant in fold['test']]
        if len(all_test) != 1040 or len(set(all_test)) != 1040:
            raise RuntimeError('Outer test splits do not cover 1,040 participants exactly once')
        for fold in splits['folds']:
            sets = [set(fold[key]) for key in ('proper_train', 'calibration', 'test')]
            if any(len(group) != len(fold[key]) for group, key in zip(sets, ('proper_train', 'calibration', 'test'))):
                raise RuntimeError('Duplicate participant in outer split')
            if sets[0] & sets[1] or (sets[0] | sets[1]) & sets[2]:
                raise RuntimeError('Outer proper-train/calibration/test participants overlap')
            if any(set(split['train']) & set(split['validation']) for split in fold['inner_folds']):
                raise RuntimeError('Inner train/validation participants overlap')
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(resolved, cfg)
        atomic_json(output / 'candidate_grid.json', cfg['frozen_candidates'])
        atomic_json(output / 'nested_splits.json', splits)
        metric_rows, latency_frames = [], []
        for fold in splits['folds']:
            folder = output / f'outer_{fold["outer_fold"]:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            inner = _inner_search(table, fold, frozen_candidates(), cfg, folder)
            _inner_blend(fold, inner, folder, cfg)
            selected = {'full': _select(inner, 'full')['candidate']['id'],
                        'clinical_full': 'clinical_full_beats_cb_baseline'}
            atomic_json(folder / 'selection.json', {'winners': selected, 'rule': cfg['selection_rule'],
                        'test_labels_read': False, 'classification': 'post-hoc exploratory'})
            base_results = {key: value for key, value in inner.items() if key != BLEND['id']}
            bundles = _fit_reported(table, fold, base_results, cfg, folder, REPORT)
            bundles[BLEND['id']] = _fit_blend(table, fold, inner[BLEND['id']], bundles, cfg, folder)
            metric_rows.extend(_evaluate(table, fold, bundles, selected, cfg, folder))
            definition = xai.explain_fold(bundles, table, fold, cfg, folder)
            atomic_json(folder / 'xai_definition.json', definition)
            latency, _ = _measure_latency(table, manifest, fold, bundles, cfg, folder)
            if not latency.empty:
                latency_frames.append(latency)
            for bundle in bundles.values():
                models.release(bundle['model'])
            print(f'Outer fold {fold["outer_fold"] + 1}/{len(splits["folds"])} complete', flush=True)
        metrics = pd.DataFrame(metric_rows)
        _compare_fixed_baselines(reference, metrics)
        metrics.to_csv(output / 'outer_fold_metrics.csv', index=False)
        metrics.groupby(['candidate', 'endpoint', 'policy'])[
            ['raw_roc_auc', 'roc_auc', 'pr_auc', 'uar', 'sensitivity', 'specificity', 'brier', 'coverage']
        ].agg(['mean', 'std']).to_csv(output / 'mean_std_metrics.csv')
        latency = pd.concat(latency_frames, ignore_index=True)
        latency.to_csv(output / 'inference_latency.csv', index=False)
        latency.groupby('candidate').audio_to_classifier_seconds.agg(
            p50_seconds='median', p95_seconds=lambda values: values.quantile(.95)
        ).to_csv(output / 'inference_latency_summary.csv')
        selected_ids = [json.loads((output / f'outer_{index:02d}' / 'selection.json').read_text())['winners']['full']
                        for index in range(10)]
        chosen = metrics[(metrics.endpoint == 'full') & (metrics.policy == 'youden')]
        selected_rows = pd.concat([chosen[(chosen.outer_fold == index) & (chosen.candidate == candidate)]
                                   for index, candidate in enumerate(selected_ids)])
        base = metrics[(metrics.candidate == 'full_beats_cb_baseline') & (metrics.policy == 'youden')].sort_values('outer_fold')
        clin = metrics[(metrics.candidate == 'clinical_full_beats_cb_baseline') & (metrics.policy == 'youden')].sort_values('outer_fold')
        differences = pd.DataFrame({'outer_fold': range(10),
            'full_beats_baseline_minus_clinical_raw_auc': base.raw_roc_auc.to_numpy() - clin.raw_roc_auc.to_numpy(),
            'selected_full_minus_clinical_raw_auc': selected_rows.sort_values('outer_fold').raw_roc_auc.to_numpy() - clin.raw_roc_auc.to_numpy()})
        differences.to_csv(output / 'paired_raw_auc_differences.csv', index=False)
        selected_mean = float(selected_rows.raw_roc_auc.mean())
        summary = {'participants': 1040, 'audio_clips': 9281, 'outer_folds': 10, 'inner_folds': 5,
                   'candidate_ids': [candidate['id'] for candidate in cfg['frozen_candidates']],
                   'classification': 'post-hoc exploratory; outer-test informed candidate design',
                   'selected_full_raw_roc_auc_mean': selected_mean,
                   'selected_full_raw_roc_auc_sd': float(selected_rows.raw_roc_auc.std()),
                   'strict_target': 0.81, 'strict_target_exceeded': selected_mean > .81,
                   'fixed_full_baseline_raw_roc_auc_mean': float(base.raw_roc_auc.mean()),
                   'fixed_clinical_baseline_raw_roc_auc_mean': float(clin.raw_roc_auc.mean()),
                   'fixed_full_minus_clinical_mean': float(base.raw_roc_auc.mean() - clin.raw_roc_auc.mean()),
                   'folds_completed': int(metrics.outer_fold.nunique()),
                   'five_reported_groups_xai_complete': all(
                       len(pd.read_csv(output / f'outer_{i:02d}' / 'grouped_xai.csv').candidate.unique()) == 5
                       for i in range(10))}
        atomic_json(output / 'run_summary.json', summary)
    print(f'Full audio-clinical exploratory run complete: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--reference-run', required=True, type=Path)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    run(args.project_root, args.reference_run, args.run_name)


if __name__ == '__main__':
    main()
