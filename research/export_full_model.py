"""Export the selected fold-zero full audio-clinical model for local research."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from cpu_model_experiment import digest
from src import coda_research as cr
from src import coda_v5_models as v5
from src import coda_v6_models as models
from full_audio_clinical_experiment import BLEND


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _same_predictions(source_bundle, exported_bundle, table, participant):
    row = table.loc[[participant]]
    source_raw = models.predict_raw(source_bundle['model'], row)
    exported_raw = models.predict_raw(exported_bundle['model'], row)
    source_calibrated = source_bundle['calibrator'].predict(source_raw)
    exported_calibrated = exported_bundle['calibrator'].predict(exported_raw)
    np.testing.assert_allclose(source_raw, exported_raw, rtol=0, atol=1e-12)
    np.testing.assert_allclose(source_calibrated, exported_calibrated, rtol=0, atol=1e-12)


def export(run_dir: Path, output_dir: Path) -> Path:
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    root = PACKAGE.parent
    resolved = _json(run_dir / 'resolved_config.json')
    splits = _json(run_dir / 'nested_splits.json')
    grid = _json(run_dir / 'candidate_grid.json')
    summary = _json(run_dir / 'run_summary.json')
    selection = _json(run_dir / 'outer_00' / 'selection.json')

    expected = {'id': 'full_blend_equal', 'endpoint': 'full', 'audio': ['beats'],
                'clinical': v5.CLINICAL + ['heart_rate'], 'family': 'weighted_blend',
                'component_ids': ['full_beats_cb_baseline', 'clinical_full_beats_cb_baseline'],
                'weights': [0.5, 0.5], 'role': 'late_fusion_control'}
    folds = splits.get('folds', [])
    if (len(folds) != 10 or sorted(fold.get('outer_fold') for fold in folds) != list(range(10)) or
            resolved.get('outer_folds') != 10):
        raise ValueError('Run is not a complete 10-fold full-model run')
    if not resolved.get('protocol_fingerprint') or len(resolved['protocol_fingerprint']) != 64:
        raise ValueError('Missing protocol fingerprint')
    if selection.get('winners', {}).get('full') != expected['id']:
        raise ValueError('Fold zero did not select full_blend_equal')
    if grid[-1] != expected or resolved.get('frozen_candidates', [])[-1] != expected:
        raise ValueError('Run candidate does not match the frozen BLEND definition')
    if summary.get('candidate_ids', [])[-1:] != [expected['id']] or summary.get('folds_completed') != 10:
        raise ValueError('Run summary does not match the full-model export protocol')
    component_specs = {candidate['id']: candidate for candidate in grid}
    if not set(expected['component_ids']).issubset(component_specs):
        raise ValueError('Candidate grid is missing a blend component definition')
    fold = folds[0]
    proper = sorted(fold['proper_train'])
    calibration_ids = list(fold['calibration'])
    if fold.get('outer_fold') != 0 or set(proper) & set(calibration_ids):
        raise ValueError('Fold-zero participant partitions are invalid')

    model_dir = run_dir / 'outer_00' / 'models' / expected['id']
    source = model_dir / 'bundle.joblib'
    sidecar = model_dir / 'bundle.sha256'
    source_hash = cr.sha256(source)
    if not sidecar.is_file() or sidecar.read_text(encoding='ascii').strip() != source_hash:
        raise ValueError('Source bundle SHA-256 does not match its sidecar')

    oof_path = run_dir / 'outer_00' / 'inner' / 'full_blend_equal_oof.json'
    oof_checkpoint = _json(oof_path)
    oof = oof_checkpoint.get('oof', [])
    expected_oof_signature = digest({'protocol': resolved['protocol_fingerprint'], 'fold': fold,
                                     'candidate': expected, 'oof': oof})
    if oof_checkpoint.get('signature') != expected_oof_signature or oof_checkpoint.get('candidate') != expected:
        raise ValueError('Blend OOF checkpoint signature or candidate is invalid')
    oof_ids = [row['participant'] for row in oof]
    if len(oof_ids) != len(set(oof_ids)) or sorted(oof_ids) != proper:
        raise ValueError('Blend OOF rows do not cover exactly the proper-train participants')

    bundle = joblib.load(source)
    expected_bundle_signature = digest({'protocol': resolved['protocol_fingerprint'], 'fold': fold,
                                        'candidate': expected, 'oof_participants': proper})
    if bundle.get('signature') != expected_bundle_signature or bundle.get('candidate') != expected:
        raise ValueError('Source bundle signature or candidate is invalid')
    if sorted(bundle.get('proper_ids', [])) != proper or set(bundle.get('calibration_ids', [])) != set(calibration_ids):
        raise ValueError('Bundle participant metadata does not match fold zero')
    component_ids = bundle.get('model', {}).get('components', {})
    if set(component_ids) != set(expected['component_ids']):
        raise ValueError('Blend components do not match the frozen candidate')
    for component_id, component in component_ids.items():
        if component.get('candidate') != component_specs[component_id]:
            raise ValueError('Blend component model metadata is inconsistent')
    if set(bundle.get('thresholds', {})) != {'youden', 'sensitivity90'}:
        raise ValueError('Bundle is missing a required threshold policy')
    if len(bundle.get('calibration_ids', [])) != len(bundle.get('calibration_raw', [])) or len(bundle.get('calibration_ids', [])) != len(bundle.get('calibration_probability', [])):
        raise ValueError('Bundle calibration metadata lengths differ')

    try:
        source_run = os.path.relpath(run_dir, root)
    except ValueError:
        source_run = str(run_dir)
    manifest = {
        'format': 1,
        'source_run': source_run,
        'candidate_id': expected['id'],
        'selected_fold': 0,
        'protocol_fingerprint': resolved['protocol_fingerprint'],
        'source_sha256': source_hash,
        'export_sha256': source_hash,
        'clinical_fields': expected['clinical'],
        'audio_inputs': expected['audio'],
        'threshold_policies': sorted(bundle['thresholds']),
        'proper_train_count': len(proper),
        'calibration_count': len(calibration_ids),
        'claim_limit': 'Fold-zero model fit on proper-train only; exploratory research artifact, not clinical validation or a score for one exported model.'
    }
    if output_dir.exists():
        target_bundle = output_dir / 'bundle.joblib'
        target_sha = output_dir / 'bundle.sha256'
        target_manifest = output_dir / 'manifest.json'
        if (not target_bundle.is_file() or not target_sha.is_file() or not target_manifest.is_file() or
                cr.sha256(target_bundle) != source_hash or
                target_sha.read_text(encoding='ascii').strip() != source_hash or
                _json(target_manifest) != manifest):
            raise FileExistsError(f'Export destination exists and differs; left unchanged: {output_dir}')
        exported = joblib.load(target_bundle)
        table = joblib.load(run_dir / 'participant_features.joblib')
        participant = calibration_ids[0]
        _same_predictions(bundle, exported, table, participant)
        return output_dir

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output_dir.name}.', dir=output_dir.parent))
    try:
        copied = temporary / 'bundle.joblib'
        shutil.copyfile(source, copied)
        if cr.sha256(copied) != source_hash:
            raise IOError('Copied bundle SHA-256 differs from source')
        (temporary / 'bundle.sha256').write_text(source_hash, encoding='ascii')
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        exported = joblib.load(copied)
        table = joblib.load(run_dir / 'participant_features.joblib')
        participant = calibration_ids[0]
        _same_predictions(bundle, exported, table, participant)
        if output_dir.exists():
            raise FileExistsError(f'Export destination appeared during export: {output_dir}')
        os.rename(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(export(args.run_dir, args.output_dir))


if __name__ == '__main__':
    main()
