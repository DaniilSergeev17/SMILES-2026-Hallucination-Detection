# SOLUTION

## Short version

The probe is a small MLP - one hidden layer, 896 to 256 hidden_dims: the hidden state at the **last token of layer 14** of Qwen2.5-0.5B. The features get a `RobustScaler` and then I clip the scaled values to +-5. To stop the per-fold numbers from bouncing around I train 11 of these MLPs with different seeds and average their outputs. The data is split 5-fold, grouped by the source context paragraph, and the decision threshold is tuned for accuracy.

On that grouped split it lands at roughly **0.72–0.73 accuracy**, against **0.70** for just always predicting "hallucinated". The `results.json` produced by `solution.py` shows 0.7229.

That is not a thrilling number, and the reason is worth stating up front: this dataset is SQuAD-style - a context paragraph, a question, the model's answer - and many questions reuse the same paragraph. The stock `splitting.py` does a plain stratified split, which leaks the paragraph across train and test, so the stock code *looks* like it gets approximately 0.73–0.76. Once you split by paragraph instead, almost nothing clears the majority class by more than a couple of points, and the layer-14 + robust-scaling MLP is the thing that does.

I ran many experiments. The full table is in `runs/summary.csv`. Nothing past the layer-14 MLP-ensemble winner actually pushed the accuracy higher - the alt-probe runs that followed were a "creative tries" pass, and the next idea I tried was a batch of regularization tricks (AWP / R-Drop / EMA / SoftAUC / Model Soups) wrapped around each net of the same 11-net ensemble.

## How to reproduce

```bash
pip install -r requirements.txt
python solution.py
```

That writes `results.json` and `predictions.csv` into the repo root.

A few things that matter for getting the same numbers:

- Only `aggregation.py`, `probe.py` and `splitting.py` were changed. `model.py`, `evaluate.py` and `solution.py` are exactly as shipped.
- The MLP training is seeded, so a run is deterministic on a given machine. bf16 arithmetic is slightly different on CPU vs Apple MPS vs CUDA, which nudges the features and therefore the final number by about half a point. I saw 0.72–0.73 on every machine I ran it on (H100 and CPU).
- `solution.py` fits the final probe with `fit()` and never calls `fit_hyperparameters()`, so out of the box the submitted probe would sit at a 0.5 decision threshold. I made `HallucinationProbe.fit()` pick an accuracy-maximizing threshold at the end of fitting. This does not change the cross-validated metrics in `results.json` - the evaluation loop calls `fit_hyperparameters()` on the validation fold, which overrides it - it only sets the threshold used on the actual `test.csv` predictions.

## What the final solution is

**`aggregation.py`.** `aggregate()` returns the hidden state at the last real (non-padding) token of layer 14 - index 14 of the 25-element `hidden_states` tuple, where index 0 is the embeddings and 1–24 are the transformer layers. That's the whole feature vector: 896 dim size. No geometric features, no concatenation of layers, no mean-pooling over tokens. `extract_geometric_features` is a no-op. `USE_GEOMETRIC` is hardcoded to `False` in `solution.py` anyway, so when I wanted to try geometric features during experiments I appended them inside `aggregate()` directly - but they didn't help, so the shipped version doesn't.

**`probe.py`.** `HallucinationProbe`:
- preprocessing is `RobustScaler` (centre on the median, scale by the IQR) followed by clipping every value to the range +-5;
- the classifier is 11 copies of `Linear(896, 256) => ReLU => Linear(256, 1)`, each trained from a different seed for 200 full-batch Adam steps with `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`; `predict_proba` averages the 11 sigmoids;
- `fit_hyperparameters()` sweeps the decision threshold to maximize accuracy on whatever validation set it gets - the evaluation loop hands it the validation fold, and `fit()` calls it in-sample at the end so the submitted probe doesn't stay at 0.5.

**`splitting.py`.** `StratifiedGroupKFold`, 5 folds, grouped on a hash of the context paragraph pulled out of the `prompt` string. The original `train_test_split` is still there as a `single` mode, but grouped k-fold is the default.

## Why these choices, and what actually moved the metric

Roughly in order of how much it mattered:

**Layer 14 instead of the last layer, plus robust scaling and clipping.** This is the main thing, and it's a bit embarrassing. If you take the stock probe - last token of layer 24, `StandardScaler`, a plain MLP - and just point it at an intermediate layer, it doesn't learn. It can't even fit the *training* set; it sits at the majority-class accuracy. The cause is that the per-token hidden states at intermediate layers have a few "massive activation" dimensions - values in the hundreds while everything else is order-1. `StandardScaler` divides each feature by its standard deviation, but those dimensions still carry most of the variance, the first linear layer is dominated by them, and 200 Adam steps get nowhere. Swap in `RobustScaler` and clip to +-5 and the problem goes away - layer 14 trains normally and generalizes a little better than the last layer. That is the jump from about 0.70 (i.e. the majority class) to about 0.72. A short sweep over layers with this recipe put 14 (and 20) on top; the embedding-adjacent layers and the final layer are a hair worse.

**Grouping the split by paragraph.** This one *lowered* my cross-validation number rather than raising it, which is the point. With a stratified random split the baseline reads approximately 0.73–0.76 and looks strong; group by paragraph and it drops to approximately 0.705, basically the majority class. So everything is run on grouped 5-fold and approximately 0.72 is what I treat as real. A side benefit: with k-fold, `solution.py`'s "fit the final probe on all the non-test data" step ends up using all 689 rows.

**Averaging 11 MLPs and tuning the threshold for accuracy.** Worth maybe a point over a single MLP with the default F1-tuned threshold, and most of that is the threshold metric: the contest scores accuracy, the stock code tunes the threshold for F1, and with classes around 70/30 those are not the same threshold. The 11-net average mostly just smooths the per-fold numbers. Going to 21 nets did nothing more; a wider hidden layer did nothing more.

So: approximately 0.70 => approximately 0.72–0.73, and a fair chunk of the "improvement over the stock code's reported number" is really just no longer leaking the split.

## What didn't work

A lot. The grouped-CV column is the honest one and the majority baseline there is 0.701.

From the early runs:

- **Geometric / topological features** - per-layer activation norms, cosine similarity between consecutive layers, an EigenScore-style log-eigenvalue spectrum of the token covariance, a PCA participation-ratio "intrinsic dimension" proxy, response length. Tried alongside the dense features and on their own. Configs with them sat around 0.68, *below* the majority class. Probably, the signal isn't there at 0.5B.
- **Concatenating several layers** - last tokens, or mean+last, of 3–6 layers. Always slightly worse than one good layer. With 480 training rows per fold you do not want 5,000–11,000 features.
- **Mean-pooling over tokens** instead of the last token - worse on layer 14. The last token already aggregates the sequence through attention.
- **HistGradientBoosting** - I expected a scale-invariant tree model to sidestep the massive-activation problem completely. It did, and it scored 0.67. The useful signal looks like a linear direction in feature space, not something trees split on cleanly.
- **A difference-of-class-means probe** (the "geometry of truth" idea) - 0.67. Too blunt.
- **Logistic regression with PCA, with ANOVA-F feature selection, bagged** - all variants, all around 0.69, none of them beat the plain layer-14 MLP. `LogisticRegressionCV` (regularization chosen by inner-CV AUROC) on layer 14 with robust scaling got to near 0.711, which is fine, just a touch below the MLP.
- **A heavily regularized MLP** - dropout 0.4, weight decay 1e-3, 3,000 epochs with early stopping. Approximately 0.66. Over-regularized; the plain over-fitting MLP, which hits 0.94 train accuracy, generalizes slightly better here. Counterintuitive, but it held across folds.
- **Class balancing** (`class_weight="balanced"` or a balancing `pos_weight`) on the linear probes - pushed accuracy *below* 0.70. With a 70/30 prior and a weak signal, "balanced" is the wrong objective; you trade correct majority predictions for wrong minority ones. (The MLP keeps `pos_weight` but it overfits enough that it isn't really balanced in practice)
- **10-fold CV** - just noisier; smaller test folds, worse variance estimate.
- **Bigger ensembles, wider nets, more epochs** - flat.

The alt-probe runs were a deliberately wider net of "creative" tries on top of the layer-14 MLP-ensemble winner - mixup, label smoothing, SWA, a 1-D conv over the layer trajectory, hand-crafted trajectory features, an SE-gated linear probe, ExtraTrees, kNN, and a stacking ensemble. None of them set a new high on accuracy, and most of them dropped it. The interesting wrinkle is that what *did* improve was AUROC: ExtraTrees at 0.724, the trajectory features at 0.716, and the stacking ensemble at **0.725** - clearly better at *ranking* hallucinated answers above truthful ones than the MLP family (0.68). None of them converted that to accuracy, because the MLP's polarized, slightly over-confident outputs hand the threshold-sweep a more cooperative distribution. The metric for this contest is accuracy, so the polarized MLP ships; if it were AUROC I'd ship the stacking ensemble.

The next idea I tried was a wider feature-and-probe pass on the same paragraph-grouped 5-fold split - response-token-tail pooling (mean of the last 32 real tokens) at layers 13/14/15, multi-layer concatenations of 12+13, 13+16, 12+16+20 and a wide 17–24 mean, PCA => logistic regression, an RBF SVM on the layer-14 vector, and ANOVA-F top-512 => `LogisticRegressionCV`. All of them landed in the same 0.69–0.72 accuracy band, with 17–24 mean + PCA + LR collapsing to **0.656**. My read on that collapse: response-tail pooling and wide late-layer concatenations look like they help on a stratified random split mostly because they let the probe latch onto the paragraph identity, and the moment you group the split by paragraph the apparent gain goes away. The closest of these feature recipes was actually the simplest one - layer-14 last token => PCA(128) => balanced LR at 0.717 - which is just my pipeline minus the MLP, and KBest => `LogisticRegressionCV` did the same trick on the AUROC side at **0.7205**, the highest AUROC across these new runs.

The regularization runs - AWP (epsilon=1e-3), R-Drop (KL between two dropout passes, alpha=1.0), an EMA shadow at alpha=0.999, Model Soups (uniform weight average across the 11 ensemble members instead of the prediction average), a SoftAUC pairwise surrogate added to BCE, and a combined EMA + R-Drop + SoftAUC run, all applied per-net inside the existing 11-net ensemble - were more mixed. **AWP** at epsilon=1e-3 was the closest to keeping the baseline intact, at **0.7243** vs the layer-14 winner's 0.7315; not an improvement, but the only one that didn't visibly hurt accuracy. SoftAUC was a wash (0.7141) and gave nothing on AUROC. EMA, R-Drop and Model Soups all dropped to 0.69–0.70. The combined regularization run had middling accuracy (0.7128) but the best AUROC of these regularization tries (0.7107), which fits the same alt-probe pattern of "ranks well, doesn't convert". Model Soups is also more or less a duplicate of what the 11-net seed average already does, and explicit weight-averaging *hurts* it, which makes sense - averaging weights only helps when the models share a loss basin, and 11 random seeds clearly don't.

## Did anything beat the baseline?

It depends what "baseline" means here.

On the *stock split* - one stratified train/val/test - the stock probe gets 0.73–0.76 and almost nothing I did beats it on that same split. Most of my changes make it *worse* there, because they target generalization and that split doesn't really test generalization.

On the *honest* grouped split, the stock probe falls to 0.705 accuracy - it sits about where the majority class sits, though oddly it keeps one of the better AUROCs I tried, 0.71, so its ranking is fine and its accuracy just isn't. The layer-14 + robust-scaling + MLP-ensemble family clearly beats that on accuracy, 0.72–0.73. So yes - on the metric that matters once you stop leaking the split.

One honest wrinkle: the config I'm submitting has the best *accuracy* but not the best *AUROC*. The accuracy edge partly comes from the threshold. If the contest scored AUROC I'd ship the stacking ensemble at 0.725, or ExtraTrees at 0.724, or the trajectory-features + linear probe at 0.716. It scores accuracy, so I'm shipping the one with the best accuracy.

## What I'd try next

Three things stand out after many experiments.

The AUROC-vs-accuracy gap from the alt-probe and regularization runs still says the lever to push past 0.73 is *calibration / threshold selection*, not another probe architecture. The stacking ensemble and the combined-regularization MLP both rank better than the shipped probe but cap out at ordinary accuracy because the threshold lands in a noisy 70/30 distribution. Isotonic regression on out-of-fold probabilities, or directly optimising a smooth accuracy surrogate during training, are the natural next steps.

AWP was the only regularization trick that didn't visibly hurt accuracy - 0.7243 against the layer-14 winner's 0.7315. I only tried epsilon=1e-3 with a single perturbation step inside each of the 11 ensemble members. A small grid over epsilon and a 2-step variant is probably worth one more pass; the gap to the plain ensemble is under one point, so a tuned AWP could realistically erase it.
