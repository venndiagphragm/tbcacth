# TBCatch research

This directory contains the current V5 experiment notebooks, frozen model/research code, asset provenance and verification tests. Earlier standalone notebooks, package-build scripts and validation runs are archived outside the active project. V3/V4 Python modules that remain are transitive dependencies or are protected by the V5 source hashes; do not delete them based on their version names.

## Inputs and layout

```text
research/
  CODA_TB_V5_Research.ipynb       EDA, preprocessing, nested CV and evaluation
  CODA_TB_V5_Resume_XAI.ipynb     Postprocessing-only XAI repair
  config.json, config_v5.json    Original research configuration
  src/                          Frozen source and vendored BEATs code
  prepare_v5_assets.py           Explicit pretrained-asset preparation
  assets_v5.json                 Pinned asset hashes and source provenance
  tools/audit_dataset.py         Read-only dataset audit
  pretrained/v5/                Local weights (ignored)
  wheels_v5/                    Local offline additions (ignored)
  references/                   Local comparison/split references (ignored)
  runs/                         Generated runs and participant results (ignored)
  model_bank_v5/                Local imported trained artifacts (ignored)
  reports/                      Local audit reports/evidence (ignored)
data/
  dataset/dataset/               Authorized original CODA files (ignored)
  dataset-audit/local-full/      Audit manifest and outputs (ignored)
```

Data are not redistributed. Obtain source access and honor data-use terms. The notebook discovers `research/` and defaults its project input root to the sibling `data/` directory. `PACKAGE_DIR` may explicitly select this research directory; `PROJECT_DIR` must select the directory containing `dataset/` and `dataset-audit/` (normally `<project>/data`).

## Environment and running

Use Python 3.11, the tested torch/torchaudio 2.6.0+cu118 pair, tabpfn 2.0.9 and librosa 0.10.2.post1. Install the matching PyTorch build for the training machine first, then the pinned lists in `requirements.txt` and `requirements-v5-additions.txt` using the appropriate PyTorch wheel index or authorized local wheel collection. There is no automatic network fallback in the current notebook; its bootstrap uses local `wheels_v5/` only for missing additions. It does not replace PyTorch.

Pretrained weights are ignored in Git. Restore them with the recorded hashes or explicitly run `prepare_v5_assets.py` after reviewing source/access/license terms. It may require internet and authorized model access. Existing local assets and offline additions remain on this laptop.

Open `CODA_TB_V5_Research.ipynb`. Use `MODE="preflight"` on this laptop. Full training requires the training GPU and respects the 45 GB VRAM limit; never claim that using more memory guarantees better performance. Select a new RUN_NAME for a new scientific configuration. Resume only with matching code, input and environment fingerprints. Notebook outputs and execution counts are cleared for Git; original executed notebooks and returns remain in the external archive/private local storage.

The original research source/config files were preserved byte-for-byte during reorganization. Folder discovery and the notebook's input-root argument changed; no fold, threshold, model-selection rule or archived experiment result was changed. `test_v5_pipeline.py` is an optional synthetic/model-parity check, not TB training; it can take time on CPU. `test_v5_xai_resume.py` covers pointwise XAI repair. Both require research dependencies and assets.

## CPU experiment track

The V6 CPU track is separate from the frozen V5 source and does not run TabPFN. It uses 10 participant-disjoint outer folds, 5 inner folds and a 15% calibration split. Candidate selection uses inner ROC-AUC, then PR-AUC, then candidate ID. Outer-test labels are reserved for evaluation.

Create a Python 3.11 environment and install the CPU-only dependencies:

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe --index https://download.pytorch.org/whl/cpu torch==2.6.0 torchaudio==2.6.0
uv pip install --python .venv/Scripts/python.exe -r research/requirements-cpu.txt
```

`research/tools/fetch_cpu_weights.py` downloads the pinned BEATs checkpoint from [mooneyko/BEATs at commit `18cfdf9d43820b2db86c6dfde4ae2f7531d1f5ad`](https://huggingface.co/mooneyko/BEATs/tree/18cfdf9d43820b2db86c6dfde4ae2f7531d1f5ad) and verifies its recorded size and SHA-256. The mirror's LFS object matches `assets_v5.json`; it is not authenticated against the publisher.

Do not fetch or train on CODA until its terms and intended use authorize the work. Once authorized files are present in `data/dataset/dataset/`, run `research/tools/audit_dataset.py`, then `research/tools/build_cpu_manifest.py`. Start a run only when `data/dataset-audit/local-full/solicited_cpu_manifest.csv` exists:

```powershell
.venv/Scripts/python.exe research/cpu_model_experiment.py --project-root data --run-name cpu_model_001
```
The regression suite is synthetic only: `.venv/Scripts/python.exe -m unittest discover -s research -p test_cpu_model_experiment.py -q`. It does not estimate TB performance or validate phone recordings.

Runs, checkpoints, caches, model bundles and participant outputs stay local and are ignored by Git. This is an exploratory CPU protocol, not the published 10×5 reproduction. It uses the local solicited cohort, BEATs embeddings and the audited audio policy; cohort size and equivalence must be established from authorized data. The app remains unchanged until evidence, missing-input behavior and a valid non-cough detector are available.

### Full audio-clinical exploratory rerun

`full_audio_clinical_experiment.py` evaluates the fixed BEATs baseline, historical full-model control, paper-descriptor alternative, equal-weight BEATs/clinical score blend, and clinical-only control against the immutable `cpu_model_available_data_001` cohort and splits:

```powershell
.venv/Scripts/python.exe research/full_audio_clinical_experiment.py --project-root data --reference-run research/runs/cpu_model_available_data_001 --run-name full_audio_clinical_004
```

The runner validates reference hashes and split identity before model fitting. Its design is post-hoc exploratory because earlier outer-test results informed the candidates; it is not an independent confirmation. The strict >0.81 comparison uses the selected full-endpoint raw ROC-AUC mean, not clinical-only or a single fold. The earlier run and audio-only/audio+HR endpoints remain unchanged.

Export the selected fold-zero `full_blend_equal` bundle for local full-input research:

```powershell
.venv/Scripts/python.exe research/export_full_model.py --run-dir research/runs/full_audio_clinical_004 --output-dir research/.local/models/active_full
```

The exporter verifies run metadata, selection, OOF coverage, source checksum, bundle signature and prediction parity, then writes a local manifest. The stable path `research/.local/models/active_full/bundle.joblib` is for research code that accepts all 16 clinical/heart-rate fields plus BEATs features. It is a fold-zero model trained on proper-train participants only, not the cross-validation mean or a clinical deployment model. Existing destinations are never overwritten. The nine-answer web prototype remains on its separate V5 bundle.

## Research vs prototype

V5 compares full clinical, paper-style audio, frozen BEATs, TabPFN/CatBoost/LR, limited ensembles and phone-available inputs, with participant-disjoint nested evaluation and XAI. The lightweight web app in `../prototype/` uses separate fixed BEATs + CatBoost candidates and an app-only threshold. Its results are not the research winner's performance. No external Indonesia validation or physical-phone pulse validation is established.

Method and historical context: `V5_README.md`, `V5_XAI_FIX_README.md`. Local original return: `../.local/artifacts/experiment_v5_001_RETURN.zip`. Do not commit individual-level exports, raw audio, fitted models or model caches. Refer to `../THIRD_PARTY_NOTICES.md` for third-party licensing boundaries.
