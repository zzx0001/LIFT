# LIFT 

Reference implementation of the LIFT plugin used in the paper.

## Contents


| File                      | Role                                                                                                                                                                             |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lift_core.py`            | Channel-aware Semantics Embedding, Time-Dependent Continuous Gabor Tokenizer, and the Frequency Trust Mask (contribution*i*).                                                    |
| `lift_plugin.py`          | `LIFTFrequencyPlugin` that composes the embedding, tokenizer, and pooling into a single plug-in module, plus a small MLP head (`LIFTPluginHead`) used by the LIFT-only ablation. |
| `safe_fusion.py`          | `UncertaintyAwareSafeFusion`: zero-initialized confidence-gated logit residual that preserves the base classifier at initialization (contribution *ii*).                         |
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


