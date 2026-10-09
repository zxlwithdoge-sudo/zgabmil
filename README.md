# z-GABMIL: depth-aware multiple-instance learning for OCT of brain tissue

Attention-based MIL for classifying optical coherence tomography (OCT) frames of
post-mortem brain tissue (AD vs NC, CTE vs NC), extended along the axis that
whole-slide methods do not have: **depth**.

An OCT volume is a stack of consecutive frames through the same tissue. On this
cohort that third axis carries more redundancy than the in-plane one:

| Neighbour relation | Feature cosine | Excess over shuffled |
|---|---:|---:|
| same (row, col), Δframe = 1 (**z**) | 0.988 | **+0.071** |
| adjacent patch, same frame (in-plane) | 0.953 | +0.026 |

This repository contains the model, the training code, the evaluation, and the
attribution tooling used to test whether that redundancy is exploitable.

## What is here

```
abmil_oct/
  model.py           ABMIL (gated attention) + GABMIL's Spatial Information
                     Mixing Module: BLOCK (local window) and GRID (dilated)
  zgabmil.py         ZSpatialMixer / GABMILZ - SIMM with a third, z stage
  volume_data.py     frame bags served with their +/- w neighbouring frames
  train_zgabmil.py   training loop for GABMILZ (mirrors train.py exactly)
  zsmooth_eval.py    zero-parameter z smoothing of attention, inference only
  additive.py        Additive MIL head (ablation; see Results)
  contributions.py   exact signed per-patch class contributions + faithfulness
  render_maps.py     attention vs contribution overlays for chosen frames
  data.py train.py evaluate.py features.py preprocess.py ...
tests/               72 tests, including the z-module correctness guards
scripts/             fold ablation runners
```

## The z stage

GABMIL's SIMM mixes patch *features* on two in-plane axes. `ZSpatialMixer` adds
a third, built exactly like the other two - a channel-shared linear map over
positions plus a residual:

```
dense -> [BLOCK] -> [GRID] -> [Z] -> attention -> pool -> classify
```

What deliberately does **not** change, so results stay comparable to the ABMIL
baseline and the GABMIL ablation:

* the bag is still one frame; neighbouring frames enter only as context,
* the attention softmax is still over one frame's patches,
* the loss is still per frame and the subject prediction a majority vote.

Only the target frame's features are altered, by information from its
neighbours. `z_window=0` reproduces the original per-frame `SpatialMixer`
bit-for-bit (`tests/test_zgabmil.py`).

### One detail that matters

Coordinates must come from `data.absolute_grid_coordinates`. The default
per-frame convention re-origins every frame separately; on this cohort that
misaligns 4-13% of adjacent frame pairs by one patch and silently destroys the
correspondence the z stage exists to use. `OCTContextBags` handles this.
**An earlier negative z result was entirely an artefact of this bug** - six of
ten folds changed the sign of their effect once it was fixed.

## Results on this cohort (10 subject-level folds)

### AD vs NC

| | Frame AUC | Folds improved | p |
|---|---:|---:|---:|
| ABMIL baseline | 0.758 ± 0.184 | — | — |
| GABMIL in-plane (best variant) | 0.726 | 0/10 | — |
| trained z-GABMIL, w = 5 | 0.770 | 6/10 | 0.57 |
| **zero-parameter z smoothing, w = 5** | **0.780** | **9/10** | **0.027** |

Read this honestly: **the trained module loses to the free operator it was
built to generalise.** The operator adds no parameters and needs no retraining;
its gain is monotone in the window size, saturates around w = 5-8, and the
pre-specifiable test is the per-fold slope of AUC on w (mean +0.0043, positive
in 9/10, t p = 0.031, Wilcoxon p = 0.020). None of the individual per-window
comparisons survives Holm correction.

In-plane GABMIL *hurts*, with a clean dose-response in window size - which is
the point: the axis GABMIL does not use is the one carrying the signal.

### CTE vs NC

Baseline 0.633 frame AUC on the current folds. z5 vs none is +0.010 (4/10
folds, p = 0.85) - nothing. Note that the fold assignment alone moved this task
from 0.461 to 0.633, so treat CTE conclusions with care.

### Additive MIL head (ablation, not adopted)

A non-linear per-patch predictor (Javed et al., NeurIPS 2022) was evaluated on
both tasks. Frame AUC changed by −0.003 on AD/NC (4/10 folds, p = 0.77) and
−0.077 on CTE/NC (2/10 folds, p = 0.13), while test loss rose in both (+0.54
and +1.13, Wilcoxon p = 0.010 on CTE/NC) against a training AUC pushed to
1.000. The linear head is retained.

The classifier being linear means the bag score *already* admits the exact
additive decomposition, with no architectural change and no loss of
expressivity: measured max |classical − additive| = 1.49e-07. `contributions.py`
exploits that.

## Install

```bash
pip install -r requirements.txt        # or: conda env create -f environment.yml
pytest tests/ -q                       # 72 tests
```

## Use

Zero-parameter z smoothing on an existing checkpoint (no retraining):

```bash
python -m abmil_oct.zsmooth_eval \
  --results-dir results/ad_nc/none \
  --features-dir abmil_oct/data/features_112_masked \
  --windows 0 1 2 3 5
```

`--windows 0` reproduces the stock model exactly and is the correctness check:
it must match the checkpoint's own `subject_summary.json`.

Training the z module:

```bash
python -m abmil_oct.train_zgabmil \
  --features-dir abmil_oct/data/features_112_masked \
  --splits-path abmil_oct/splits/AD_NC_folds.json \
  --results-dir results/ad_nc_z5 \
  --label0 NC --label1 AD \
  --folds 10 --epochs 12 \
  --learning-rate 2e-4 --weight-decay 1e-5 --dropout 0.25 \
  --spatial-mode none --z-window 5 --seed 0
```

Exact per-patch attribution and the attention-vs-contribution overlays:

```bash
python -m abmil_oct.contributions --checkpoint <fold.pt> \
  --features-dir <features> --output-dir <out>

python -m abmil_oct.render_maps --checkpoint <fold.pt> \
  --features-dir <features> --patches-dir <patches> --frames-dir <frames> \
  --fold-split test --output-dir <out>
```

`render_maps` draws attention as `alpha / alpha_max` (never a percentile rank,
which is uniform by construction and hides how concentrated attention really
is) and the contribution as a **signed percentage of the frame's total
evidence**, on a diverging symlog scale with zero neutral.

## Reading the attribution maps

Attention is one unsigned, class-agnostic scalar per patch. A contribution is
signed and per class and sums exactly to the bag logits. On this cohort they
disagree sharply: attention correlates 0.90 with |contribution| but **−0.31**
with the signed contribution, and 70% of above-median-attention patches argue
*against* the frame's own prediction.

Attention is also extremely concentrated - median effective patch count 3.4 out
of ~253 with the linear head (9.0 with the additive head). On many frames a
single patch carries most of the decision.

**The maps say which patches drove the model, not which tissue is diseased.**
Without region-level ground truth those are different claims, and on this
cohort the top-attention occupancy pattern identifies the *subject* at 0.976
(chance 0.250).

## Caveats

13 subjects. Four test subjects per fold, so subject accuracy is quantised to
{0, .25, .5, .75, 1} and the 10-fold mean moves in steps of 0.025 - frame-level
AUC is the metric with resolution. Fold 5 shows AUC 0.989 against frame
accuracy 0.735, i.e. a near-perfect ranking with a badly placed decision
threshold; calibration is worth more here than any architecture change tested
so far.

## References

- Ilse, Tomczak & Welling, *Attention-based Deep Multiple Instance Learning*, ICML 2018.
- Keshvarikhojasteh et al., *GABMIL*, arXiv:2504.17379 — the in-plane SIMM this work extends.
- Javed et al., *Additive MIL: Intrinsically Interpretable Multiple Instance Learning for Pathology*, NeurIPS 2022.

## License

See `LICENSE`.
