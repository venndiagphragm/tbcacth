"""Run a frozen participant-disjoint CPU comparison on an authorized CODA manifest."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from src import coda_research as cr
from src import coda_v4 as v4
from src import coda_v5_features as features
from src import coda_v6_models as models
from src import coda_v6_xai as xai


REPORT_IDS = {
    'audio_only': ['audio_only_beats_cb_baseline', 'audio_only_paper_cb', 'audio_only_beats_lr_c01'],
    'audio_hr': ['audio_hr_beats_cb_baseline', 'v5_beats_hr_02', 'audio_hr_beats_lr_c01'],
    'mobile': ['mobile_beats_cb_baseline', 'v5_beats_mobile_02'],
    'full': ['full_beats_cb_baseline', 'v5_beats_full_cb_01'],
    'clinical_full': ['clinical_full_beats_cb_baseline'],
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def atomic_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _fingerprint(path):
    return cr.sha256(path)


def configure(project_root, run_name, check_existing=True):
    if not run_name.isascii() or not run_name.replace('_', '').replace('-', '').isalnum() or len(run_name) > 71:
        raise ValueError('Invalid run name')
    root = Path(project_root).resolve()
    data_root = root / 'dataset' / 'dataset'
    manifest = root / 'dataset-audit' / 'local-full' / 'solicited_cpu_manifest.csv'
    if not data_root.is_dir() or not manifest.is_file():
        raise FileNotFoundError(f'Authorized audited dataset and solicited_cpu_manifest.csv are required under {root}')
    for relative in ['meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv',
                     'meta_data/Clinical/CODA_TB_additional_variables_train.csv']:
        if not (data_root / relative).is_file():
            raise FileNotFoundError(data_root / relative)
    assets = json.loads((PACKAGE / 'assets_v5.json').read_text(encoding='utf-8'))['assets']
    beats_asset = next(asset for asset in assets
                       if asset['path'] == 'pretrained/v5/BEATs_iter3_plus_AS2M.pt')
    beats_path = PACKAGE / beats_asset['path']
    if not beats_path.is_file():
        raise FileNotFoundError(f'Run research/tools/fetch_cpu_weights.py first: {beats_path}')
    if beats_path.stat().st_size != beats_asset['bytes'] or _fingerprint(beats_path) != beats_asset['sha256']:
        raise ValueError(f'Pinned BEATs size/SHA-256 mismatch: {beats_path}')
    cfg = cr.load_config(PACKAGE)
    cfg.update(json.loads((PACKAGE / 'config_v5.json').read_text(encoding='utf-8')))
    cfg.update(data_root=str(data_root), manifest=str(manifest), package_root=str(PACKAGE),
               output_dir=str(PACKAGE / 'runs' / run_name), mode='cpu', require_cuda_for_full=False,
               tabpfn_device='cpu', beats_batch_size=2, cpu_threads=4, feature_workers=2,
               outer_folds=10, inner_folds=5, outer_train_calibration_fraction=0.15,
               seed=42, calibration_fraction=0.15, selection_rule='mean_inner_roc_auc_then_pr_auc_then_candidate_id',
               target_sensitivity=0.9, gate_calibration_target=0.95, xai_per_outer_fold=4,
               xai_background=8, beats_precision='float32')
    cfg['input_hashes'] = {
        'manifest': _fingerprint(manifest),
        **{p.name: _fingerprint(p) for p in (data_root / 'meta_data/Clinical').glob('*.csv')},
        'beats_checkpoint_sha256': beats_asset['sha256'],
    }
    source_paths = [PACKAGE / 'config_v5.json', PACKAGE / 'assets_v5.json',
                    PACKAGE / 'src/coda_research.py', PACKAGE / 'src/coda_v4.py',
                    PACKAGE / 'src/coda_v5_features.py', PACKAGE / 'src/coda_v5_models.py',
                    PACKAGE / 'src/coda_v6_models.py', PACKAGE / 'src/coda_v6_xai.py',
                    PACKAGE / 'cpu_model_experiment.py']
    cfg['source_hashes'] = {p.relative_to(PACKAGE).as_posix(): _fingerprint(p) for p in source_paths}
    cfg['versions'] = {name: metadata.version(name) for name in
                       ['torch', 'torchaudio', 'numpy', 'pandas', 'scipy', 'scikit-learn',
                        'catboost', 'librosa', 'joblib']}
    cfg['protocol_fingerprint'] = digest({key: value for key, value in cfg.items()
                                         if key not in ['output_dir', 'package_root']})
    out = Path(cfg['output_dir'])
    resolved = out / 'resolved_config.json'
    if check_existing and resolved.exists():
        previous = json.loads(resolved.read_text(encoding='utf-8'))
        if previous.get('protocol_fingerprint') != cfg['protocol_fingerprint']:
            raise RuntimeError('Run fingerprint differs; use a new --run-name. Existing checkpoints remain untouched.')
    return cfg


def _checkpoint(path, signature, producer):
    path = Path(path)
    checksum = path.with_suffix(path.suffix + '.sha256')
    if path.exists() or checksum.exists():
        if not path.is_file() or not checksum.is_file() or cr.sha256(path) != checksum.read_text().strip():
            raise RuntimeError(f'Checkpoint is incomplete or changed: {path}')
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('signature') != signature:
            raise RuntimeError(f'Checkpoint fingerprint mismatch: {path}')
        return value
    value = producer()
    atomic_json(path, value)
    checksum.write_text(cr.sha256(path), encoding='ascii')
    return value


def _finite_scores(y, p):
    mask = np.isfinite(p)
    y = np.asarray(y, dtype=int)[mask]
    p = np.asarray(p, dtype=float)[mask]
    if len(np.unique(y)) != 2:
        raise ValueError('Candidate has fewer than two labels among eligible participants')
    return mask, {'roc_auc': float(roc_auc_score(y, p)),
                  'pr_auc': float(average_precision_score(y, p)), 'n': int(mask.sum())}


def _inner_search(table, fold, candidates, cfg, folder):
    records = []
    results = {}
    signature = digest({'protocol': cfg['protocol_fingerprint'], 'fold': fold})
    for candidate in candidates:
        candidate_rows = []
        eligible_ids = set()
        for inner_index, split in enumerate(fold['inner_folds']):
            checkpoint = Path(folder) / 'inner' / f'{candidate["id"]}_{inner_index}.json'
            key = digest({'signature': signature, 'candidate': candidate, 'split': split})

            def train():
                started = time.perf_counter()
                model = models.fit_model(table, split['train'], candidate, cfg,
                                         cfg['seed'] + 1000 * fold['outer_fold'] + inner_index)
                try:
                    validation = table.loc[split['validation']]
                    raw = models.predict_raw(model, validation)
                finally:
                    models.release(model)
                labels = validation.tb_status.to_numpy(dtype=int)
                _, metric = _finite_scores(labels, raw)
                return {'signature': key, 'candidate': candidate,
                        'train_ids': list(split['train']), 'validation_ids': list(split['validation']),
                        'y': labels.tolist(), 'raw_probability': raw.tolist(), 'metrics': metric,
                        'seconds': time.perf_counter() - started}

            result = _checkpoint(checkpoint, key, train)
            if result['candidate'] != candidate or result['train_ids'] != split['train'] or result['validation_ids'] != split['validation']:
                raise RuntimeError(f'Inner-fold identity mismatch: {checkpoint}')
            expected = table.loc[split['validation'], 'tb_status'].astype(int).tolist()
            if result['y'] != expected:
                raise RuntimeError(f'Cached labels no longer match: {checkpoint}')
            raw_values = np.asarray(result['raw_probability'], dtype=float)
            eligible_ids.update(pid for pid, probability in zip(result['validation_ids'], raw_values)
                                if np.isfinite(probability))
            candidate_rows.extend({'participant': pid, 'y': label, 'raw_probability': probability,
                                   'inner_fold': inner_index}
                                  for pid, label, probability in zip(result['validation_ids'], result['y'], raw_values)
                                  if np.isfinite(probability))
        frame = pd.DataFrame(candidate_rows)
        if frame.participant.duplicated().any() or set(frame.participant) != eligible_ids:
            raise RuntimeError(f'Inner-fold participant coverage mismatch: {candidate["id"]}')
        fold_auc = []; fold_pr = []
        for inner_index in range(len(fold['inner_folds'])):
            part = frame[frame.inner_fold == inner_index]
            _, metrics = _finite_scores(part.y.to_numpy(), part.raw_probability.to_numpy())
            fold_auc.append(metrics['roc_auc']); fold_pr.append(metrics['pr_auc'])
        results[candidate['id']] = {'candidate': candidate, 'oof': frame,
                                    'mean_inner_roc_auc': float(np.mean(fold_auc)),
                                    'mean_inner_pr_auc': float(np.mean(fold_pr)),
                                    'coverage': int(len(frame))}
        records.append({key: value for key, value in results[candidate['id']].items() if key != 'oof'})
        pd.DataFrame(records).to_csv(Path(folder) / 'inner_search.csv', index=False)
    return results


def _select(results, endpoint):
    candidates = [result for result in results.values() if result['candidate']['endpoint'] == endpoint]
    return sorted(candidates, key=lambda row: (-row['mean_inner_roc_auc'],
                                                -row['mean_inner_pr_auc'], row['candidate']['id']))[0]


def _reported(results, report_ids=REPORT_IDS):
    selected = {}
    for endpoint, ids in report_ids.items():
        winner = _select(results, endpoint)['candidate']['id']
        for candidate_id in [winner, *ids]:
            if candidate_id in results:
                selected[candidate_id] = results[candidate_id]
    return selected


def _fit_reported(table, fold, results, cfg, folder, report_ids=REPORT_IDS):
    bundles = {}
    for candidate_id, result in _reported(results, report_ids).items():
        candidate = result['candidate']
        oof = result['oof']
        signature = digest({'protocol': cfg['protocol_fingerprint'], 'fold': fold,
                            'candidate': candidate, 'oof_participants': oof.participant.tolist()})
        path = Path(folder) / 'models' / candidate_id / 'bundle.joblib'
        sidecar = path.with_suffix('.sha256')
        if path.exists() or sidecar.exists():
            if not path.is_file() or not sidecar.is_file() or cr.sha256(path) != sidecar.read_text().strip():
                raise RuntimeError(f'Model checkpoint is incomplete or changed: {path}')
            bundle = joblib.load(path)
            if bundle.get('signature') != signature:
                raise RuntimeError(f'Model checkpoint fingerprint mismatch: {path}')
        else:
            proper_oof = oof.sort_values('participant')
            if len(proper_oof) < 2 or proper_oof.y.nunique() != 2:
                raise ValueError(f'Insufficient complete OOF rows for calibration: {candidate_id}')
            calibrator = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip')
            calibrator.fit(proper_oof.raw_probability, proper_oof.y)
            model = models.fit_model(table, fold['proper_train'], candidate, cfg,
                                     cfg['seed'] + 1000 * fold['outer_fold'] + 99)
            calibration = table.loc[fold['calibration']]
            try:
                raw = models.predict_raw(model, calibration)
                mask = np.isfinite(raw)
                cal = calibration.loc[mask]
                calibrated = np.asarray(calibrator.predict(raw[mask]), dtype=float)
                if cal.tb_status.nunique() != 2:
                    raise ValueError('Calibration participants do not include both labels')
                thresholds = {policy: v4.threshold(cal.tb_status.to_numpy(int), calibrated, policy,
                                                   cfg['target_sensitivity']) for policy in ['youden', 'sensitivity90']}
                bundle = {'model': model, 'calibrator': calibrator, 'candidate': candidate,
                          'signature': signature, 'thresholds': thresholds,
                          'proper_ids': list(fold['proper_train']), 'calibration_ids': cal.index.tolist(),
                          'calibration_raw': raw[mask].tolist(), 'calibration_probability': calibrated.tolist()}
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.part')
                joblib.dump(bundle, temporary, compress=3)
                temporary.replace(path)
                sidecar.write_text(cr.sha256(path), encoding='ascii')
            except BaseException:
                models.release(model)
                raise
            bundle['model'] = model
        bundles[candidate_id] = bundle
    return bundles


def _metrics(y, probabilities, threshold):
    mask = np.isfinite(probabilities)
    y = np.asarray(y, dtype=int)[mask]
    p = np.asarray(probabilities, dtype=float)[mask]
    return {'n': int(len(y)), 'coverage': float(mask.mean()),
            **v4.calculate_metrics(y, p, threshold)}


def _evaluate(table, fold, bundles, selected, cfg, folder):
    rows = []; predictions = []
    test = table.loc[fold['test']]
    for candidate_id, bundle in bundles.items():
        raw = models.predict_raw(bundle['model'], test)
        valid = np.isfinite(raw)
        probability = np.full(len(test), np.nan, dtype=float)
        probability[valid] = bundle['calibrator'].predict(raw[valid])
        y = test.tb_status.to_numpy(dtype=int)
        _, auc = _finite_scores(y, probability)
        for policy, cutoff in bundle['thresholds'].items():
            m = _metrics(y, probability, cutoff)
            rows.append({'outer_fold': fold['outer_fold'], 'candidate': candidate_id,
                         'endpoint': bundle['candidate']['endpoint'], 'policy': policy,
                         'raw_roc_auc': float(roc_auc_score(y[valid], raw[valid])),
                         'pr_auc': auc['pr_auc'], **m, 'threshold': cutoff,
                         'selected': selected[bundle['candidate']['endpoint']] == candidate_id})
        for pid, label, p, r in zip(test.index, y, probability, raw):
            if not np.isfinite(p):
                continue
            predictions.append({'outer_fold': fold['outer_fold'], 'candidate': candidate_id,
                                'endpoint': bundle['candidate']['endpoint'], 'participant': pid,
                                'y': int(label), 'probability': float(p), 'raw_probability': float(r)})
    pd.DataFrame(rows).to_csv(Path(folder) / 'outer_test_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(Path(folder) / 'outer_test_predictions.csv', index=False)
    return rows


def freeze_cascade(raw_gate_scores, labels, missing_hr, mobile_probabilities, cfg):
    raw_gate = v4.threshold(labels, raw_gate_scores, 'sensitivity90', .95)
    routed = route_questionnaire(raw_gate_scores, missing_hr, raw_gate)
    filters_negative = bool((~routed[np.asarray(labels, dtype=int) == 0]).any())
    gate = raw_gate if filters_negative else None
    mobile_cut = v4.threshold(labels, mobile_probabilities, 'sensitivity90', cfg['target_sensitivity'])
    return {'stage1_raw_gate': gate, 'proposed_raw_gate': raw_gate,
            'gate_filters_calibration_negative': filters_negative,
            'stage2_calibrated_cutoff': float(mobile_cut)}


def route_questionnaire(raw_audio_score, missing_hr, gate):
    raw_audio_score = np.asarray(raw_audio_score, dtype=float)
    missing_hr = np.asarray(missing_hr, dtype=bool)
    if gate is None:
        return np.ones(len(raw_audio_score), dtype=bool)
    return (raw_audio_score >= gate) | missing_hr


def _evaluate_cascade(table, fold, bundles, selected, cfg, folder):
    audio_bundle = bundles[selected['audio_only']]
    mobile_bundle = bundles[selected['mobile']]
    calibration = table.loc[fold['calibration']]
    raw_cal = models.predict_raw(audio_bundle['model'], calibration)
    if not np.isfinite(raw_cal).all():
        raise RuntimeError('Audio-only gate unexpectedly abstained on calibration participants')
    labels_cal = calibration.tb_status.to_numpy(dtype=int)
    hr_missing_cal = calibration.heart_rate.isna().to_numpy()
    mobile_cal = mobile_bundle['calibrator'].predict(models.predict_raw(mobile_bundle['model'], calibration))
    thresholds = freeze_cascade(raw_cal, labels_cal, hr_missing_cal, mobile_cal, cfg)
    gate = thresholds['stage1_raw_gate']
    proposed_gate = thresholds['proposed_raw_gate']
    filters_negative = thresholds['gate_filters_calibration_negative']
    stage2_cut = thresholds['stage2_calibrated_cutoff']
    test = table.loc[fold['test']]
    raw_test = models.predict_raw(audio_bundle['model'], test)
    routed = route_questionnaire(raw_test, test.heart_rate.isna().to_numpy(), gate)
    mobile_test = mobile_bundle['calibrator'].predict(models.predict_raw(mobile_bundle['model'], test))
    decision = routed & (mobile_test >= stage2_cut)
    labels = test.tb_status.to_numpy(dtype=int)
    tp = int(((labels == 1) & decision).sum())
    fn = int(((labels == 1) & ~decision).sum())
    tn = int(((labels == 0) & ~decision).sum())
    fp = int(((labels == 0) & decision).sum())
    atomic_json(Path(folder) / 'cascade_frozen.json', {
        'stage1_candidate': selected['audio_only'], 'stage1_raw_gate': gate,
        'proposed_raw_gate': proposed_gate, 'gate_filters_calibration_negative': filters_negative,
        'stage2_candidate': selected['mobile'], 'stage2_calibrated_cutoff': float(stage2_cut),
        'thresholds_fit_on': 'calibration participants only',
        'missing_hr_routes_to_questionnaire': True,
    })
    pd.DataFrame({'participant': test.index, 'y': labels, 'questionnaire_required': routed,
                  'prediction': decision.astype(int), 'stage1_raw_score': raw_test,
                  'stage2_probability': mobile_test}).to_csv(Path(folder) / 'cascade_predictions.csv', index=False)
    return {'outer_fold': fold['outer_fold'], 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
            'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp),
            'questionnaire_fraction': float(routed.mean()), 'gate': gate,
            'stage2_cutoff': float(stage2_cut),
            'stage1_missed_tb': int(((labels == 1) & ~routed).sum())}


def _measure_latency(table, manifest, fold, bundles, cfg, folder):
    ids = sorted(np.random.default_rng(cfg['seed'] + 4000 + fold['outer_fold'])
                 .choice(fold['test'], min(cfg['xai_per_outer_fold'], len(fold['test'])), replace=False).tolist())
    audio = manifest[manifest.participant.isin(ids)].sort_values('filename').groupby('participant').head(1)
    audio = audio.set_index('participant')
    needs_beats = any('beats' in bundle['candidate']['audio'] for bundle in bundles.values())
    load_seconds = None
    net = None
    if needs_beats:
        started = time.perf_counter()
        net = features.load_beats(cfg, 'cpu')
        load_seconds = time.perf_counter() - started
    extracted = {}
    try:
        for pid in ids:
            if pid not in audio.index:
                continue
            path = Path(cfg['data_root']) / audio.loc[pid, 'path']
            started = time.perf_counter()
            wave = features.read_physical(path)
            decode_seconds = time.perf_counter() - started
            values = {}
            for modality in {'beats' for b in bundles.values() if 'beats' in b['candidate']['audio']} | {
                    'paper' for b in bundles.values() if 'paper' in b['candidate']['audio']}:
                started = time.perf_counter()
                if modality == 'beats':
                    values[modality] = features.embed_batch(net, [wave], cfg, 'cpu')
                else:
                    values[modality] = features.paper_features(wave)[None, :]
                values[f'{modality}_seconds'] = time.perf_counter() - started
            extracted[pid] = (values, decode_seconds)
        rows = []
        for candidate_id, bundle in bundles.items():
            model_path = Path(folder) / 'models' / candidate_id / 'bundle.joblib'
            started = time.perf_counter()
            cold = joblib.load(model_path)
            cold_load = time.perf_counter() - started
            for pid, (values, decode_seconds) in extracted.items():
                candidate = bundle['candidate']
                if candidate['family'] == 'logistic_regression' and candidate['clinical']:
                    if not np.isfinite(table.loc[pid, candidate['clinical']].to_numpy(dtype=float)).all():
                        continue
                single = table.loc[[pid]].copy()
                feature_seconds = 0.0
                for modality in candidate['audio']:
                    single.at[pid, f'_bag_{modality}'] = values[modality]
                    feature_seconds += values[f'{modality}_seconds']
                started = time.perf_counter()
                models.predict_raw(cold['model'], single)
                predict_seconds = time.perf_counter() - started
                rows.append({'outer_fold': fold['outer_fold'], 'candidate': candidate_id,
                             'participant': pid, 'decode_seconds': decode_seconds,
                             'feature_seconds': feature_seconds, 'classifier_seconds': predict_seconds,
                             'audio_to_classifier_seconds': decode_seconds + feature_seconds + predict_seconds,
                             'cold_model_load_seconds': cold_load, 'encoder_cold_load_seconds': load_seconds,
                             'model_bundle_bytes': model_path.stat().st_size,
                             'encoder_checkpoint_bytes': (Path(PACKAGE / cfg['beats_checkpoint']).stat().st_size
                                                          if 'beats' in candidate['audio'] else 0),
                             'detector_included': False, 'recordings_per_participant': 1,
                             'source': 'single CODA solicited recording'})
            del cold
        frame = pd.DataFrame(rows)
        frame.to_csv(Path(folder) / 'inference_latency.csv', index=False)
        return frame, load_seconds
    finally:
        if net is not None:
            del net


def run(project_root, run_name):
    cfg = configure(project_root, run_name)
    out = Path(cfg['output_dir'])
    cr.seed_everything(cfg['seed'])
    import torch
    torch.set_num_threads(cfg['cpu_threads'])
    with threadpool_limits(limits=cfg['cpu_threads']):
        participants, manifest = v4.audit_cohort(cfg)
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'resolved_config.json', cfg)
        atomic_json(out / 'candidate_grid.json', models.candidate_grid())
        table = features.prepare_features(participants, manifest, cfg, include_beats=True)
        splits = v4.nested_splits(participants, cfg)
        candidates = models.candidate_grid()
        candidates.sort(key=lambda candidate: candidate['role'] == 'tuning')
        summaries = []
        latency_frames = []
        cascade_rows = []
        for fold in splits['folds']:
            folder = out / f'outer_{fold["outer_fold"]:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            inner = _inner_search(table, fold, candidates, cfg, folder)
            selected = {endpoint: _select(inner, endpoint)['candidate']['id'] for endpoint in models.INPUTS}
            atomic_json(folder / 'selection.json', {'winners': selected,
                        'rule': cfg['selection_rule'], 'test_labels_read': False})
            bundles = _fit_reported(table, fold, inner, cfg, folder)
            summaries.extend(_evaluate(table, fold, bundles, selected, cfg, folder))
            xai_definition = xai.explain_fold(bundles, table, fold, cfg, folder)
            atomic_json(folder / 'xai_definition.json', xai_definition)
            latency_frame, encoder_load_seconds = _measure_latency(table, manifest, fold, bundles, cfg, folder)
            if not latency_frame.empty:
                latency_frame['encoder_cold_load_seconds'] = encoder_load_seconds
                latency_frames.append(latency_frame)
            cascade_rows.append(_evaluate_cascade(table, fold, bundles, selected, cfg, folder))
            for bundle in bundles.values():
                models.release(bundle['model'])
            print(f'Outer fold {fold["outer_fold"] + 1}/{len(splits["folds"])} complete', flush=True)
        metrics = pd.DataFrame(summaries)
        metrics.to_csv(out / 'outer_fold_metrics.csv', index=False)
        aggregations = metrics.groupby(['candidate', 'endpoint', 'policy'])[
            ['roc_auc', 'pr_auc', 'uar', 'sensitivity', 'specificity', 'brier', 'coverage']].agg(['mean', 'std'])
        aggregations.to_csv(out / 'mean_std_metrics.csv')
        pd.DataFrame(cascade_rows).to_csv(out / 'cascade_outer_metrics.csv', index=False)
        latency = pd.concat(latency_frames, ignore_index=True)
        latency.to_csv(out / 'inference_latency.csv', index=False)
        latency_summary = latency.groupby('candidate').audio_to_classifier_seconds.agg(
            p50_seconds='median', p95_seconds=lambda values: values.quantile(.95))
        latency_summary.to_csv(out / 'inference_latency_summary.csv')
        audit = {'participants': len(participants), 'audio_clips': len(manifest),
                 'split_unit': 'participant', 'outer_folds': cfg['outer_folds'],
                 'inner_folds': cfg['inner_folds'], 'calibration_fraction': cfg['outer_train_calibration_fraction'],
                 'test_coverage_by_candidate': metrics.groupby('candidate').coverage.mean().to_dict(),
                 'detector': 'not trained: authorized non-cough negative recordings not established',
                 'journal_protocol_difference': 'Local solicited subset, BEATs embeddings and reviewed-audio policy may differ from 1,105-participant journal cohort.',
                 'latency_scope': 'One solicited CODA recording per participant; excludes unavailable cough detector and is not a phone benchmark.'}
        chosen = metrics[(metrics.selected) & (metrics.policy == 'youden')]
        targets = {}
        for endpoint, cutoff in [('audio_only', .70), ('audio_hr', .70), ('full', .81)]:
            scores = chosen.loc[chosen.endpoint == endpoint, 'raw_roc_auc']
            targets[endpoint] = {'target_mean_roc_auc': cutoff, 'mean_outer_roc_auc': float(scores.mean()),
                                'sd_outer_roc_auc': float(scores.std()), 'target_exceeded': bool(scores.mean() > cutoff)}
        audit['roc_auc_targets'] = targets
        atomic_json(out / 'run_summary.json', audit)
    print(f'CPU experiment complete: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    run(args.project_root, args.run_name)


if __name__ == '__main__':
    main()
