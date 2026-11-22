# Wafer Map Pattern Classification: Stacking Handcrafted and Convolutional Features, Revisited

Code, logs and results for the paper *"Stacking Handcrafted and Convolutional Features for Wafer Map Pattern
Classification, Revisited: Reproduction, Meta-Training Pitfalls, and Improvements"* (`research_paper/paper/main.pdf`).

It reproduces Kang & Kang, *Computers in Industry* 129 (2021) 103450 (`research_paper/reference/journal.pdf`), on WM-811K and extends it.

| | Test macro-F1 |
|---|---|
| Kang & Kang (2021), Stacking-MLR | 0.8949 ± 0.0121 |
| Our reproduction, Stacking-MLR | 0.8967 ± 0.0060 |
| **Our final pipeline** (5 CNNs with test-time augmentation + XGBoost → MLR) | **0.9134** |

With a confidence-based reject option, the final pipeline classifies 95.2% of test wafers automatically at 99.72% accuracy.

## Layout
```
src/                 the pipeline: common.py (shared helpers) + one runnable script per step (table below)
dataset/             raw WM-811K dataset, LSWMD.pkl (local-only, download instructions in dataset/README.md)
data/                preprocessed inputs (X_MFE.pkl, y.pkl; X_CNN.pkl is local-only) and every saved model output (*.pkl)
models/              saved models (XGBoost, MFE-FNN, MLR + calibration in final_pipeline.npz; CNN weights local-only)
logs/                raw trial logs (*.jsonl), training logs, GPU queue scripts
results/             result tables (*.md) and figures/ - generated from data/ and logs/
research_paper/      publication material:
  paper/             IEEE-format paper: main.tex, main.pdf, refs.bib; make_assets.py builds every table and figure
  reference/         journal.pdf - the reproduced paper (Kang & Kang, 2021)
third_party/         the authors' reference code: WMPC_Stacking_TF2 (feature extraction used by src/phase2_data.py)
                     and wafermap_MultiNN (git submodule; reference only)
archive/             first attempt (phase3-7 scripts, notebooks) and one-off fix scripts - superseded by src/
run_cpu.sh run_gpu.sh setup_env.sh   environment setup and launchers
```

| Step | Script (`src/`) | Reads → writes |
|---|---|---|
| 1. Preprocess | `phase2_data.py` | `dataset/LSWMD.pkl` → `data/X_CNN.pkl`, `data/X_MFE.pkl` (59 handcrafted features), `data/y.pkl` |
| 2. Base learners | `mfe_tune.py`, `cnn_train.py` (`--tta`, `--multinn`, `--subset`), `xgb_tune.py`, `average_outputs.py` | `data/X_*.pkl` → `data/*_outputs.pkl`, `models/` |
| 3. Stacking | `stack_tune.py` (MLR, FNN, DT; selection on split B, `final=1` for test) | `data/*_outputs.pkl` → `data/stack_*.pkl`, `logs/stack_trials.jsonl` |
| 4. Evaluation | `compare.py` (paired t-test + bootstrap), `calibrate.py` (temperature + reject option), `nsweep.py` (training-size sweep) | → `logs/*.jsonl`, `data/calib_*.pkl`, `results/` |
| 5. Explainability | `mlr_weights.py`, `gradcam.py`, `shap_xgb.py` | → `results/figures/` |
| 6. Reports | `freeze_results.py`, `extensions_results.py`, `research_paper/paper/make_assets.py` | `data/`, `logs/` → `results/*.md`, `research_paper/paper/tables/`, `research_paper/paper/figures/` |
| 7. Inference | `predict.py` (`verify=1` checks against the saved results; `export=1` writes `models/final_pipeline.npz`) | `models/` + test set → `data/final_predictions.pkl` |

`watch_training.py` shows live training progress from `logs/`. All paths are resolved from the repository root
(`common.P`), so scripts work from any working directory.

## Reproducing
1. Environment (Linux, NVIDIA GPU): `sh setup_env.sh` (TensorFlow 2.17 + tf_keras, XGBoost, scikit-learn).
   `run_gpu.sh` / `run_cpu.sh` run a script in that environment (`WAFER_ENV=/path/to/env` overrides it).
2. Data: download `LSWMD.pkl` (WM-811K) to `dataset/LSWMD.pkl`, then `sh run_cpu.sh src/phase2_data.py`.
3. Rebuild the tables from the saved outputs (no training): `python src/freeze_results.py`,
   `python src/extensions_results.py`, `python research_paper/paper/make_assets.py`.
4. Paper: `cd research_paper/paper && tectonic main.tex` (or pdfLaTeX + BibTeX).

Example training command:
`sh run_gpu.sh src/cnn_train.py --tag imnet_cw0_aug --imagenet --cw 0 --fp32 --aug --seed 0`.
Trained CNN weights (~170 MB each) exceed GitHub's file limit and are not included; retraining takes about two hours
each on one GPU.

## Acknowledgments
Feature extraction follows the authors' public code (`DMkelllog/WMPC_Stacking_TF2`, `DMkelllog/wafermap_MultiNN`).
