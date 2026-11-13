"""Interventional grouped Shapley explanations for V6 probability pipelines."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from . import coda_v6_models as models


def grouped_shapley(bundle, table, participant, background, cfg):
    candidate = bundle['candidate']
    groups = {}
    if candidate['audio']:
        groups['audio'] = [f'_bag_{name}' for name in candidate['audio']]
    clinical = candidate['clinical']
    non_hr = [column for column in clinical if column != 'heart_rate']
    if non_hr:
        groups['clinical_values'] = non_hr
        groups['clinical_missingness'] = non_hr
    if 'heart_rate' in clinical:
        groups['heart_rate_values'] = ['heart_rate']
        groups['heart_rate_missingness'] = ['heart_rate']
    keys = list(groups)
    if not keys:
        raise ValueError('Explanation model has no input groups')
    patient = table.loc[participant]
    paths = list(itertools.product([0, 1], repeat=len(keys)))
    frames = []
    for bits in paths:
        frame = background.copy()
        for group, active in zip(keys, bits):
            if not active:
                continue
            if group == 'audio':
                for column in groups[group]:
                    value = patient[column]
                    frame[column] = pd.Series([value] * len(frame), index=frame.index, dtype=object)
            elif group.endswith('_values'):
                for column in groups[group]:
                    value = patient[column]
                    if pd.notna(value):
                        frame[column] = float(value)
            else:
                for column in groups[group]:
                    if pd.isna(patient[column]):
                        frame[column] = np.nan
        frames.append(frame)
    batched = pd.concat(frames, ignore_index=True)
    raw = models.predict_raw(bundle['model'], batched)
    calibrated = bundle['calibrator'].predict(raw).reshape(len(paths), len(background)).mean(axis=1)
    values = dict(zip(paths, calibrated))
    direct_raw = float(models.predict_raw(bundle['model'], table.loc[[participant]])[0])
    full_raw = raw.reshape(len(paths), len(background))[-1]
    raw_error = float(np.max(np.abs(full_raw - direct_raw)))
    if raw_error > 1e-5:
        raise RuntimeError('Grouped explanation batch differs from direct prediction')
    direct_probability = float(bundle['calibrator'].predict([direct_raw])[0])
    values[(1,) * len(keys)] = direct_probability
    attributions = {key: 0.0 for key in keys}
    for index, key in enumerate(keys):
        for bits, value in values.items():
            if bits[index]:
                continue
            size = sum(bits)
            changed = list(bits)
            changed[index] = 1
            weight = math.factorial(size) * math.factorial(len(keys) - size - 1) / math.factorial(len(keys))
            attributions[key] += weight * (values[tuple(changed)] - value)
    base = values[(0,) * len(keys)]
    residual = direct_probability - base - sum(attributions.values())
    if abs(residual) > 1e-8:
        raise RuntimeError('Grouped Shapley completeness check failed')
    return {'participant': participant, 'candidate': candidate['id'],
            'base_probability': float(base), 'probability': direct_probability,
            'raw_batch_error': raw_error, 'completeness_residual': float(residual),
            **attributions}


def explain_fold(bundles, table, fold, cfg, folder):
    rng = np.random.default_rng(cfg['seed'] + 3000 + fold['outer_fold'])
    participants = sorted(rng.choice(fold['test'], min(cfg['xai_per_outer_fold'], len(fold['test'])), replace=False).tolist())
    background_ids = sorted(rng.choice(fold['proper_train'], min(cfg['xai_background'], len(fold['proper_train'])), replace=False).tolist())
    background = table.loc[background_ids]
    rows = []
    for candidate_id, bundle in bundles.items():
        for participant in participants:
            if bundle['candidate']['family'] == 'logistic_regression' and bundle['candidate']['clinical']:
                values = table.loc[participant, bundle['candidate']['clinical']].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    continue
                eligible = np.isfinite(background[bundle['candidate']['clinical']].to_numpy(dtype=float)).all(axis=1)
                local_background = background.loc[eligible]
                if len(local_background) < 2:
                    continue
            else:
                local_background = background
            rows.append(grouped_shapley(bundle, table, participant, local_background, cfg))
    frame = pd.DataFrame(rows)
    if frame.empty or frame.completeness_residual.abs().max() > 1e-8:
        raise RuntimeError('No valid grouped explanations or incomplete attribution')
    frame.to_csv(folder / 'grouped_xai.csv', index=False)
    return {'participants': participants, 'background_ids': background_ids,
            'method': 'Exact interventional Shapley over modality/clinical/HR groups; clinical missingness flags remain paired with their values.',
            'warning': 'Attributions describe this fitted model, not causal or clinical effects.'}
