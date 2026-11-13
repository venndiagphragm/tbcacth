"""Synthetic CPU contracts; this test does not evaluate real TB data."""
from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import hashlib
import sys
import wave


import joblib
from src import coda_research as cr
from src import coda_v4
from src import coda_v5_models
from src import coda_v6_models
from src import coda_v6_xai
from tools import build_cpu_manifest
import full_audio_clinical_experiment as full_audio
import export_full_model
from cpu_model_experiment import atomic_json, digest, _checkpoint, _select, freeze_cascade, route_questionnaire

class CpuModelExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.ids = [f'SYNTH_{index:03d}' for index in range(100)]
        cls.labels = np.arange(100) % 2
        signal = cls.labels + rng.normal(0, .7, len(cls.ids))
        cls.table = pd.DataFrame(index=cls.ids)
        cls.table['tb_status'] = cls.labels
        cls.table['heart_rate'] = 80 + signal * 8
        cls.table.loc[cls.ids[::11], 'heart_rate'] = np.nan
        for column in coda_v5_models.CLINICAL:
            if column == 'sex':
                cls.table[column] = (rng.integers(0, 2, len(cls.ids))).astype(float)
            else:
                cls.table[column] = signal + rng.normal(0, 1, len(cls.ids))
        cls.table.loc[cls.ids[::13], 'height'] = np.nan
        for name, width in [('beats', 12), ('paper', 8)]:
            cls.table[f'_bag_{name}'] = pd.Series(
                [rng.normal(size=(2 + index % 3, width)).astype(np.float32) + signal[index]
                 for index in range(len(cls.ids))], index=cls.ids, dtype=object)

    def _candidate(self, candidate_id):
        candidate = next(item for item in coda_v6_models.candidate_grid() if item['id'] == candidate_id)
        candidate = dict(candidate)
        candidate['params'] = {'depth': 2, 'iterations': 20, 'learning_rate': .1, 'l2_leaf_reg': 3}
        candidate['pca'] = 4
        return candidate

    def _fit(self, candidate_id):
        return coda_v6_models.fit_model(self.table, self.ids[:80], self._candidate(candidate_id),
                                        {'cpu_threads': 1}, 42)

    def _synthetic_export_run(self, root):
        run = root / 'run'
        fold = {'outer_fold': 0, 'proper_train': self.ids[:80],
                'calibration': self.ids[80:], 'test': []}
        folds = [fold]
        for index in range(1, 10):
            folds.append(dict(fold, outer_fold=index, proper_train=[], calibration=[], test=[]))
 
        candidate = dict(full_audio.BLEND)
        candidate['clinical'] = coda_v5_models.CLINICAL + ['heart_rate']
        frozen_candidates = full_audio.frozen_candidates()
        models_by_id = {}
        for candidate_id in candidate['component_ids']:
            spec = self._candidate(candidate_id)
            models_by_id[candidate_id] = coda_v6_models.fit_model(
                self.table, self.ids[:80], spec, {'cpu_threads': 1}, 42)
            frozen_candidates = [spec if item['id'] == candidate_id else item
                                 for item in frozen_candidates]
        candidates = frozen_candidates + [candidate]
        cfg = {'outer_folds': 10, 'protocol_fingerprint': 'a' * 64,
               'frozen_candidates': candidates}
        atomic_json(run / 'resolved_config.json', cfg)
        splits = {'folds': folds}
        atomic_json(run / 'nested_splits.json', splits)
        atomic_json(run / 'candidate_grid.json', candidates)
        atomic_json(run / 'run_summary.json', {
            'candidate_ids': [item['id'] for item in candidates], 'folds_completed': 10})
        atomic_json(run / 'outer_00' / 'selection.json', {'winners': {'full': candidate['id']}})
        model = {'candidate': candidate, 'components': models_by_id}
        oof = [{'participant': pid, 'raw_probability': float(index / 100),
                'y': int(self.labels[index]), 'inner_fold': index % 5}
               for index, pid in enumerate(self.ids[:80])]
        atomic_json(run / 'outer_00' / 'inner' / 'full_blend_equal_oof.json',
                    {'candidate': candidate, 'oof': oof,
                     'signature': digest({'protocol': cfg['protocol_fingerprint'], 'fold': fold,
                                          'candidate': candidate, 'oof': oof})})
        calibrator = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip')
        calibrator.fit(np.linspace(.01, .99, 80), self.labels[:80])
        bundle = {'model': model, 'calibrator': calibrator, 'candidate': candidate,
                  'signature': digest({'protocol': cfg['protocol_fingerprint'], 'fold': fold,
                                       'candidate': candidate, 'oof_participants': sorted(fold['proper_train'])}),
                  'thresholds': {'youden': {'threshold': .5}, 'sensitivity90': {'threshold': .2}},
                  'proper_ids': list(fold['proper_train']), 'calibration_ids': list(fold['calibration']),
                  'calibration_raw': [], 'calibration_probability': []}
        bundle['calibration_raw'] = coda_v6_models.predict_raw(model, self.table.loc[fold['calibration']]).tolist()
        bundle['calibration_probability'] = calibrator.predict(bundle['calibration_raw']).tolist()
        source_dir = run / 'outer_00' / 'models' / candidate['id']
        source_dir.mkdir(parents=True)
        source = source_dir / 'bundle.joblib'
        joblib.dump(bundle, source, compress=3)
        (source_dir / 'bundle.sha256').write_text(cr.sha256(source), encoding='ascii')
        joblib.dump(self.table, run / 'participant_features.joblib')
        return run, source

    def test_export_full_model_is_verified_idempotent_and_non_destructive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, source = self._synthetic_export_run(root)
            destination = root / 'active'
            self.assertEqual(export_full_model.export(run, destination), destination)
            before = (destination / 'bundle.joblib').read_bytes()
            self.assertEqual(export_full_model.export(run, destination), destination)
            self.assertEqual((destination / 'bundle.joblib').read_bytes(), before)
            manifest = json.loads((destination / 'manifest.json').read_text())
            self.assertEqual(len(manifest['clinical_fields']), 16)
            self.assertEqual(manifest['export_sha256'], cr.sha256(source))

            corrupt_run, corrupt_source = self._synthetic_export_run(root / 'corrupt')
            (corrupt_source.parent / 'bundle.sha256').write_text('0' * 64)
            with self.assertRaisesRegex(ValueError, 'sidecar'):
                export_full_model.export(corrupt_run, root / 'corrupt-active')

            wrong_run, _ = self._synthetic_export_run(root / 'wrong-candidate')
            atomic_json(wrong_run / 'outer_00' / 'selection.json', {'winners': {'full': 'wrong'}})
            with self.assertRaisesRegex(ValueError, 'did not select'):
                export_full_model.export(wrong_run, root / 'wrong-active')

            changed = json.loads((destination / 'manifest.json').read_text())
            changed['candidate_id'] = 'modified'
            (destination / 'manifest.json').write_text(json.dumps(changed))
            prior_bundle = (destination / 'bundle.joblib').read_bytes()
            with self.assertRaises(FileExistsError):
                export_full_model.export(run, destination)
            self.assertEqual((destination / 'bundle.joblib').read_bytes(), prior_bundle)

    def test_audio_only_ignores_clinical_and_hr(self):
        model = self._fit('audio_only_beats_cb_baseline')
        before = coda_v6_models.predict_raw(model, self.table.iloc[80:])
        changed = self.table.iloc[80:].copy()
        changed['heart_rate'] = -9999
        for column in coda_v5_models.CLINICAL:
            changed[column] = 1e9
        after = coda_v6_models.predict_raw(model, changed)
        np.testing.assert_array_equal(before, after)

    def test_full_catboost_keeps_natural_missing_values(self):
        model = self._fit('full_beats_cb_baseline')
        rows = self.table.iloc[80:]
        matrix, _ = coda_v6_models._design(rows, model['candidate'], model['processors'])
        self.assertTrue(np.isnan(matrix).any())
        self.assertTrue(np.isfinite(coda_v6_models.predict_raw(model, rows)).all())

    def test_missing_hr_always_continues_to_questionnaire(self):
        routed = route_questionnaire([.1, .1, .9], [False, True, False], .5)
        np.testing.assert_array_equal(routed, [False, True, True])
        np.testing.assert_array_equal(route_questionnaire([.1, .9], [False, False], None), [True, True])

    def test_test_labels_cannot_change_selection_or_calibration_cutoffs(self):
        candidates = [
            {'candidate': {'id': 'a', 'endpoint': 'audio_only'}, 'mean_inner_roc_auc': .7,
             'mean_inner_pr_auc': .6},
            {'candidate': {'id': 'b', 'endpoint': 'audio_only'}, 'mean_inner_roc_auc': .8,
             'mean_inner_pr_auc': .5},
        ]
        results = {row['candidate']['id']: row for row in candidates}
        winner_before = _select(results, 'audio_only')['candidate']['id']
        cal_y = np.tile([0, 1], 20)
        cal_raw = np.linspace(.05, .95, 40)
        mobile = np.linspace(.02, .98, 40)
        frozen_before = freeze_cascade(cal_raw, cal_y, np.zeros(40, dtype=bool), mobile,
                                       {'target_sensitivity': .9})
        test_y = np.array([0, 1, 0, 1])
        decision = np.array([False, True, False, False])
        original_counts = ((test_y == 1) & decision).sum()
        changed_counts = ((1 - test_y == 1) & decision).sum()
        winner_after = _select(results, 'audio_only')['candidate']['id']
        frozen_after = freeze_cascade(cal_raw, cal_y, np.zeros(40, dtype=bool), mobile,
                                      {'target_sensitivity': .9})
        self.assertNotEqual(original_counts, changed_counts)
        self.assertEqual(winner_before, winner_after)
        self.assertEqual(frozen_before, frozen_after)

    def test_resume_fingerprint_rejects_changed_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.json'
            _checkpoint(path, 'protocol-a', lambda: {'value': 1})
            with self.assertRaisesRegex(RuntimeError, 'fingerprint mismatch'):
                _checkpoint(path, 'protocol-b', lambda: {'value': 2})

    def test_grouped_xai_completes_to_pointwise_probability(self):
        model = self._fit('full_beats_cb_baseline')
        train = self.table.loc[self.ids[:80]]
        raw = coda_v6_models.predict_raw(model, train)
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(raw, train.tb_status)
        bundle = {'candidate': model['candidate'], 'model': model, 'calibrator': calibrator}
        explained = coda_v6_xai.grouped_shapley(bundle, self.table, self.ids[80], train.iloc[:8], {})
        direct = float(calibrator.predict(coda_v6_models.predict_raw(model, self.table.loc[[self.ids[80]]]))[0])
        self.assertAlmostEqual(explained['probability'], direct, places=12)
        self.assertAlmostEqual(explained['completeness_residual'], 0, places=10)

    def test_nested_splits_cover_each_participant_once_without_overlap(self):
        participants = self.table.reset_index(names='participant')
        with tempfile.TemporaryDirectory() as directory:
            cfg = {'outer_folds': 10, 'inner_folds': 5, 'outer_train_calibration_fraction': .15,
                   'seed': 42, 'output_dir': directory}
            splits = coda_v4.nested_splits(participants, cfg)
        test_ids = [pid for fold in splits['folds'] for pid in fold['test']]
        self.assertEqual(set(test_ids), set(self.ids))
        self.assertEqual(len(test_ids), len(set(test_ids)))
        for fold in splits['folds']:
            self.assertFalse(set(fold['proper_train']) & set(fold['calibration']))
            self.assertFalse((set(fold['proper_train']) | set(fold['calibration'])) & set(fold['test']))
            self.assertEqual(len(fold['inner_folds']), 5)


    def test_logistic_regression_abstains_on_missing_hr(self):
        candidate = next(item for item in coda_v6_models.candidate_grid()
                         if item['id'] == 'audio_hr_beats_lr_c01')
        candidate = dict(candidate)
        candidate['pca'] = 4
        model = coda_v6_models.fit_model(self.table, self.ids[:80], candidate,
                                         {'cpu_threads': 1}, 42)
        predictions = coda_v6_models.predict_raw(model, self.table.iloc[80:])
        missing = self.table.iloc[80:].heart_rate.isna().to_numpy()
        self.assertTrue(np.isnan(predictions[missing]).all())
        self.assertTrue(np.isfinite(predictions[~missing]).all())

    def test_json_checkpoints_encode_unavailable_scores_as_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.json'
            atomic_json(path, {'raw_probability': [float('nan'), .4]})
            values = np.asarray(json.loads(path.read_text())['raw_probability'], dtype=float)
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(values[1], .4)

    def test_full_paper_candidate_uses_paper_and_exact_clinical_inputs(self):
        candidate = next(item for item in full_audio.frozen_candidates() if item['id'] == 'full_paper_cb')
        self.assertEqual(candidate['audio'], ['paper'])
        self.assertEqual(candidate['clinical'], coda_v5_models.CLINICAL + ['heart_rate'])
        self.assertIsNone(candidate['pca'])
        table = self.table.copy()
        table['_bag_paper'] = pd.Series(
            [np.ones((2, 261), dtype=np.float32) for _ in self.ids], index=self.ids, dtype=object)
        matrix, _ = coda_v6_models._design(table.iloc[:8], candidate, fit=True, seed=42)
        self.assertEqual(matrix.shape, (8, 261 + 2 * len(candidate['clinical'])))

    def test_late_blend_averages_component_scores_and_xai_completes(self):
        full = self._fit('full_beats_cb_baseline')
        clinical_candidate = next(item for item in coda_v6_models.candidate_grid()
                                  if item['id'] == 'clinical_full_beats_cb_baseline')
        clinical_candidate = dict(clinical_candidate)
        clinical_candidate['params'] = {'depth': 2, 'iterations': 20, 'learning_rate': .1, 'l2_leaf_reg': 3}
        clinical = coda_v6_models.fit_model(self.table, self.ids[:80], clinical_candidate, {'cpu_threads': 1}, 43)
        blend = {'candidate': full_audio.BLEND,
                 'components': {'full_beats_cb_baseline': full,
                                'clinical_full_beats_cb_baseline': clinical}}
        rows = self.table.iloc[80:]
        expected = .5 * (coda_v6_models.predict_raw(full, rows) +
                         coda_v6_models.predict_raw(clinical, rows))
        actual = coda_v6_models.predict_raw(blend, rows)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        raw = coda_v6_models.predict_raw(blend, self.table.iloc[:80])
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(
            raw, self.table.iloc[:80].tb_status)
        bundle = {'candidate': full_audio.BLEND, 'model': blend, 'calibrator': calibrator}
        explained = coda_v6_xai.grouped_shapley(bundle, self.table, self.ids[80],
                                                self.table.iloc[:8], {})
        direct = float(calibrator.predict(actual[:1])[0])
        self.assertAlmostEqual(explained['probability'], direct, places=12)
        self.assertAlmostEqual(explained['completeness_residual'], 0, places=10)

    def test_blend_oof_rejects_missing_or_mismatched_component_rows(self):
        frame = pd.DataFrame({'participant': self.ids[:10], 'y': self.labels[:10],
                              'raw_probability': np.linspace(.1, .9, 10),
                              'inner_fold': np.repeat(range(5), 2)})
        other = frame.iloc[:-1].copy()
        results = {full_audio.BLEND['component_ids'][0]: {'oof': frame},
                   full_audio.BLEND['component_ids'][1]: {'oof': other}}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'inner').mkdir()
            pd.DataFrame([{'candidate': 'base'}]).to_csv(folder / 'inner_search.csv', index=False)
            with self.assertRaisesRegex(RuntimeError, 'Missing component OOF participant'):
                full_audio._inner_blend({'outer_fold': 0}, results, folder,
                                        {'protocol_fingerprint': 'test'})
        other = frame.copy()
        other.loc[0, 'inner_fold'] = 4
        results[full_audio.BLEND['component_ids'][1]] = {'oof': other}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'inner').mkdir()
            pd.DataFrame([{'candidate': 'base'}]).to_csv(folder / 'inner_search.csv', index=False)
            with self.assertRaisesRegex(RuntimeError, 'Mismatched label or inner_fold'):
                full_audio._inner_blend({'outer_fold': 0}, results, folder,
                                        {'protocol_fingerprint': 'test'})
    def test_outer_test_labels_cannot_change_blend_oof_or_calibration(self):
        oof = pd.DataFrame({'participant': self.ids[:40], 'y': self.labels[:40],
                            'raw_probability': np.linspace(.1, .9, 40),
                            'inner_fold': np.repeat(range(5), 8)})
        result = {'oof': oof}
        fold = {'outer_fold': 0, 'proper_train': self.ids[:40],
                'calibration': self.ids[40:60], 'test': self.ids[60:80]}
        components = {candidate_id: {'model': {}}
                      for candidate_id in full_audio.BLEND['component_ids']}
        cfg = {'protocol_fingerprint': 'synthetic', 'target_sensitivity': .9}
        with mock.patch.object(coda_v6_models, 'predict_raw',
                               side_effect=lambda model, rows: np.linspace(.05, .95, len(rows))):
            with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
                before = full_audio._fit_blend(self.table, fold, result, components, cfg, first)
                changed = self.table.copy()
                changed.loc[self.ids[60:80], 'tb_status'] = 1 - changed.loc[self.ids[60:80], 'tb_status']
                after = full_audio._fit_blend(changed, fold, result, components, cfg, second)
        self.assertEqual(before['thresholds'], after['thresholds'])
        np.testing.assert_array_equal(before['calibration_raw'], after['calibration_raw'])

    def test_manifest_builder_filters_invalid_audio_and_cross_participant_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'dataset' / 'dataset'
            audio_dir = data_root / 'raw_data' / 'solicited_data'
            audit = root / 'dataset-audit' / 'local-full'
            audio_dir.mkdir(parents=True)
            audit.mkdir(parents=True)
            inventory = []
            source = []
            samples = np.arange(8000, dtype='<i2')
            for participant, filename, offset, channels in [
                    ('A', 'a.wav', 0, 1), ('B', 'b.wav', 7, 1),
                    ('C', 'c.wav', 0, 1), ('D', 'd.wav', 13, 2), ('E', 'e.wav', 21, 1)]:
                path = audio_dir / filename
                pcm = (samples + offset).astype('<i2')
                if channels == 2:
                    pcm = np.column_stack([pcm, pcm]).ravel()
                with wave.open(str(path), 'wb') as handle:
                    handle.setnchannels(channels)
                    handle.setsampwidth(2)
                    handle.setframerate(16000)
                    handle.writeframes(pcm.tobytes())
                inventory.append({'path': path.relative_to(data_root).as_posix(), 'filename': filename,
                                  'subset': 'solicited', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'channels': channels, 'sample_width': 2, 'truncated': False,
                                  'pcm_sha256': hashlib.sha256(pcm.tobytes()).hexdigest(), 'error': ''})
                source.append({'participant': participant, 'filename': filename,
                               'tb_status': int(participant in {'C', 'D'})})
            source.append({'participant': 'F', 'filename': 'f.wav', 'tb_status': 0})
            pd.DataFrame(inventory).to_csv(audit / 'audio_inventory.csv', index=False)
            pd.DataFrame(source).to_csv(audit / 'solicited_training_manifest_unfiltered.csv', index=False)
            pd.DataFrame([{'participant': 'E', 'tb_status': 0,
                           'Microbiologicreferencestandard': 'Positive'}]).to_csv(
                               audit / 'label_review_required.csv', index=False)
            with mock.patch.object(sys, 'argv', ['build_cpu_manifest', '--project-root', str(root)]):
                build_cpu_manifest.main()
            result = pd.read_csv(audit / 'solicited_cpu_manifest.csv')
            self.assertEqual(result.participant.tolist(), ['B'])
            conflicts = pd.read_csv(audit / 'solicited_cpu_conflicts_for_review.csv')
            self.assertEqual(set(conflicts.participant), {'A', 'C', 'E'})
            exclusions = pd.read_csv(audit / 'solicited_cpu_manifest_exclusions.csv')
            self.assertEqual(set(exclusions.participant), {'D', 'F'})
            self.assertEqual(exclusions.set_index('participant').loc['F', 'reason'], 'missing_audio_file')


if __name__ == '__main__':
    unittest.main()
