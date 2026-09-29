# Model Log

Every model trained on UCI HAR, with the same protocol: a subject-wise train/val split (seed 0) and z-score
standardization fitted on train. Test = official test set (9 subjects). Metric = macro-F1. Raw rows are in
[results/metrics.csv](results/metrics.csv).

## W03 - Linear classifiers ([w03_logreg.py](experiments/part1_pre_midterm/w03_logreg.py))

| Date | Model | Setting | Split | Acc | Macro-F1 | Notes |
|---|---|---|---|---|---|---|
| 2026-09-27 | Perceptron | WALKING vs LAYING, $\eta = 0.1$, 20 ep | test | 1.0000 | 1.0000 | Separable: 0 mistakes from epoch 2 |
| 2026-09-27 | Delta rule | WALKING vs LAYING, $\eta = 0.001$, 20 ep | test | 1.0000 | 1.0000 | |
| 2026-09-27 | Logistic | WALKING vs LAYING, $\eta = 0.1$, 20 ep | test | 1.0000 | 1.0000 | |
| 2026-09-27 | Perceptron | SITTING vs STANDING | test | 0.9247 | 0.9245 | Never converges (about 100 mistakes/epoch) |
| 2026-09-27 | Delta rule | SITTING vs STANDING | test | 0.9120 | 0.9115 | |
| 2026-09-27 | Logistic | SITTING vs STANDING | test | 0.9257 | 0.9255 | |
| 2026-09-27 | Softmax regression | $\eta = 0.001$, 30 ep, $B = 64$ | val | 0.8813 | 0.8702 | Underfits |
| 2026-09-27 | Softmax regression | $\eta = 0.01$ | val | 0.9168 | 0.9099 | |
| 2026-09-27 | Softmax regression | $\eta = 0.1$ | val | 0.9284 | **0.9247** | Best on val |
| 2026-09-27 | Softmax regression | $\eta = 1.0$ | val | 0.9147 | 0.9095 | Val loss 0.758, overconfident |
| 2026-09-27 | Softmax regression | $\eta = 0.1$ (selected) | **test** | 0.9369 | **0.9365** | |
| 2026-09-27 | sklearn LogisticRegression | $C = 1.0$ (L2), train+val | test | 0.9555 | 0.9556 | Reference only |

## W04 - MLP + backprop ([w04_mlp.py](experiments/part1_pre_midterm/w04_mlp.py))

All runs use 561-128-6, 40 epochs, $B = 64$, and He init.

| Date | Model | Setting | Split | Acc | Macro-F1 | Notes |
|---|---|---|---|---|---|---|
| 2026-09-27 | MLP | ReLU, SGD $\eta = 0.01$ | val | 0.9263 | 0.9212 | Min val loss 0.195 at ep 35 |
| 2026-09-27 | MLP | ReLU, Momentum $\eta = 0.01$, $\mu = 0.9$ | val | 0.9400 | 0.9353 | Min val loss 0.184 at ep 6 |
| 2026-09-27 | MLP | ReLU, Adam $\eta = 10^{-3}$ | val | 0.9447 | 0.9428 | Min val loss 0.160 at ep 6, then overfits |
| 2026-09-27 | MLP | sigmoid, Adam $\eta = 10^{-3}$ | val | 0.9447 | 0.9437 | |
| 2026-09-27 | MLP | tanh, Adam $\eta = 10^{-3}$ | val | 0.9434 | 0.9414 | |
| 2026-09-27 | MLP | ReLU, Adam, early stop (patience 5) | **test** | 0.9471 | **0.9465** | Stopped at epoch 11 |
| 2026-09-27 | sklearn MLPClassifier | ReLU, Adam, (128,), early_stopping | test | 0.9437 | 0.9428 | Reference only |
