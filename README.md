# LIFT — Code Supplement

Reference implementation of the LIFT plugin and its data-preparation
utilities used in the paper. This folder contains the two core modelling
contributions and the dataset-preparation logic for the two clinical
tasks evaluated in the paper.

## Contents


| File                      | Role                                                                                                                                                                             |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lift_core.py`            | Channel-aware Semantics Embedding, Time-Dependent Continuous Gabor Tokenizer, and the Frequency Trust Mask (contribution*i*).                                                    |
| `lift_plugin.py`          | `LIFTFrequencyPlugin` that composes the embedding, tokenizer, and pooling into a single plug-in module, plus a small MLP head (`LIFTPluginHead`) used by the LIFT-only ablation. |
| `safe_fusion.py`          | `UncertaintyAwareSafeFusion`: zero-initialized confidence-gated logit residual that preserves the base classifier at initialization (contribution *ii*).                         |
| `masld_task_utils.py`     | MASLD progression task: loaders for MIMIC-IV and TTSH lab tables, sliding-window construction (12-month windows), stratified split, and train-statistic normalization.           |
| `mortality_task_utils.py` | 48-hour in-hospital mortality task: loaders for the pre-built MIMIC-IV and eICU npz tensors, stratified 80/10/10 split, and per-channel z-score normalization.                   |
| `environment.yml`         | Full conda environment specification.                                                                                                                                            |

## Requirements

Recreate the exact conda environment used for the paper:

```
conda env create -f environment.yml
conda activate lift
```

## Data availability

Both cohorts used in the paper are subject to the respective data-use
agreements (MIMIC-IV, eICU-CRD, and the private TTSH cohort) and are
not redistributed here.

- **MASLD** — `masld_task_utils.py` operates on a long-format CSV of
  patient-level laboratory measurements (`load_mimic_liver_source` and
  `load_ttsh_liver_source`). The expected column layout is documented
  in the source-loading functions.
- **Mortality** — `mortality_task_utils.py` reads pre-built npz tensors
  with keys `X_ts` (N × T × F_ts), `X_static` (N × F_st), `y` (N,) and
  matching feature-name arrays. Place the npz files at the paths
  configured at the top of the module (`DATA_PREPROC_DIR` and
  `EICU_DIR`).

