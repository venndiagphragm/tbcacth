"""CPU research candidates with explicit modality inputs and native clinical missingness."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .coda_v5_models import CLINICAL, MOBILE, pooled_audio

AUDIO_ONLY = []
INPUTS = {
    'audio_only': (['beats'], AUDIO_ONLY),
    'audio_hr': (['beats'], ['heart_rate']),
    'mobile': (['beats'], MOBILE + ['heart_rate']),
    'full': (['beats'], CLINICAL + ['heart_rate']),
    'clinical_full': ([], CLINICAL + ['heart_rate']),
}
TRIALS = [
    {'depth': 2, 'iterations': 400, 'learning_rate': 0.05, 'l2_leaf_reg': 10},
    {'depth': 3, 'iterations': 400, 'learning_rate': 0.05, 'l2_leaf_reg': 10},
    {'depth': 4, 'iterations': 600, 'learning_rate': 0.03, 'l2_leaf_reg': 20},
]
BASELINE = {'depth': 3, 'iterations': 600, 'learning_rate': 0.03, 'l2_leaf_reg': 10}
LEGACY = {
    'audio_hr': {'id': 'v5_beats_hr_02', 'params': BASELINE, 'pca': 32},
    'mobile': {'id': 'v5_beats_mobile_02', 'params': BASELINE, 'pca': 32},
    'full': {'id': 'v5_beats_full_cb_01',
             'params': {'depth': 4, 'iterations': 800, 'learning_rate': 0.03, 'l2_leaf_reg': 20}, 'pca': 64},
}


def candidate_grid():
    candidates = []
    for endpoint, (audio, clinical) in INPUTS.items():
        base = dict(id=f'{endpoint}_beats_cb_baseline', endpoint=endpoint,
                    audio=audio, clinical=clinical, family='catboost', params=BASELINE,
                    pca=32, role='common_baseline')
        candidates.append(base)
        if endpoint in LEGACY:
            old = LEGACY[endpoint]
            candidates.append(dict(id=old['id'], endpoint=endpoint, audio=audio,
                                   clinical=clinical, family='catboost', params=old['params'],
                                   pca=old['pca'], role='historical_parameter_control'))
        for index, params in enumerate(TRIALS, 1):
            candidates.append(dict(id=f'{endpoint}_beats_cb_trial_{index}', endpoint=endpoint,
                                   audio=audio, clinical=clinical, family='catboost',
                                   params=params, pca=32, role='tuning'))
    candidates.append(dict(id='audio_only_paper_cb', endpoint='audio_only', audio=['paper'],
                           clinical=[], family='catboost', params=BASELINE, pca=None,
                           role='light_audio_alternative'))
    for endpoint in ['audio_only', 'audio_hr']:
        audio, clinical = INPUTS[endpoint]
        candidates.append(dict(id=f'{endpoint}_beats_lr_c01', endpoint=endpoint,
                               audio=audio, clinical=clinical, family='logistic_regression',
                               C=0.1, pca=32, role='light_beats_alternative'))
    assert len({item['id'] for item in candidates}) == len(candidates)
    return candidates




def _design(table, candidate, fitted=None, fit=False, seed=42):
    blocks = []
    processors = {} if fitted is None else fitted
    if candidate['audio']:
        for modality in candidate['audio']:
            if candidate['role'] == 'light_audio_alternative':
                values = np.stack([bag.mean(axis=0, dtype=np.float64)
                                   for bag in table[f'_bag_{modality}']])
                blocks.append(values)
                continue
            values = pooled_audio(table, f'_bag_{modality}')
            key = f'audio_{modality}'
            if fit:
                steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                         ('scale', StandardScaler())]
                ncomp = min(candidate['pca'], values.shape[1], len(values) - 1)
                if ncomp > 0:
                    steps.append(('pca', PCA(n_components=ncomp, svd_solver='randomized', random_state=seed)))
                processors[key] = Pipeline(steps).fit(values)
            blocks.append(processors[key].transform(values))
    if candidate['clinical']:
        values = table[candidate['clinical']].to_numpy(dtype=np.float64, copy=False)
        missing = np.isnan(values).astype(np.float32)
        if candidate['family'] == 'logistic_regression':
            if fit:
                processors['clinical'] = StandardScaler().fit(values)
            values = processors['clinical'].transform(values)
        blocks.extend([values, missing])
    return (np.column_stack(blocks) if len(blocks) > 1 else blocks[0]), processors


def fit_model(table, ids, candidate, cfg, seed):
    rows = table.loc[list(ids)]
    if candidate['family'] == 'logistic_regression' and candidate['clinical']:
        complete = np.isfinite(rows[candidate['clinical']].to_numpy(dtype=np.float64)).all(axis=1)
        rows = rows.loc[complete]
    labels = rows.tb_status.to_numpy(dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Training fold must contain both TB labels after complete-case filtering')
    model = {'candidate': candidate, 'fit_ids': rows.index.tolist(), 'seed': seed, 'processors': {}}
    x, model['processors'] = _design(rows, candidate, model['processors'], True, seed)
    if candidate['family'] == 'catboost':
        from catboost import CatBoostClassifier
        classifier = CatBoostClassifier(**candidate['params'], random_seed=seed,
                                        thread_count=cfg['cpu_threads'], verbose=False,
                                        allow_writing_files=False, task_type='CPU')
    else:
        classifier = LogisticRegression(C=candidate['C'], max_iter=10000,
                                        random_state=seed, solver='lbfgs')
    classifier.fit(x, labels)
    model['model'] = classifier
    return model




def predict_raw(model, table):
    if model['candidate']['family'] == 'weighted_blend':
        components = model['components']
        probabilities = [predict_raw(components[candidate_id], table)
                         for candidate_id in model['candidate']['component_ids']]
        return sum(weight * probability for weight, probability
                   in zip(model['candidate']['weights'], probabilities))
    if model['candidate']['family'] == 'logistic_regression' and model['candidate']['clinical']:
        columns = model['candidate']['clinical']
        eligible = np.isfinite(table[columns].to_numpy(dtype=np.float64)).all(axis=1)
        probabilities = np.full(len(table), np.nan, dtype=float)
        if eligible.any():
            x, _ = _design(table.loc[eligible], model['candidate'], model['processors'])
            probabilities[eligible] = model['model'].predict_proba(x)[:, 1]
    else:
        x, _ = _design(table, model['candidate'], model['processors'])
        probabilities = model['model'].predict_proba(x)[:, 1]
    if not np.isfinite(probabilities[~np.isnan(probabilities)]).all():
        raise ValueError('Invalid model probabilities')
    return probabilities


def release(model):
    model.clear()
