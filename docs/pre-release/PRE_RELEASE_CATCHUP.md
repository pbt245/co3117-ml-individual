# Pre-release Catch-up (Weeks 1–4)

The individual assignment was released at the end of week 4. This note records what I covered in W01–W04, what I built during the catch-up, and where the repository stands at the `release-baseline` checkpoint.

## 1. Use case and dataset

- **Dataset:** UCI Human Activity Recognition Using Smartphones (Anguita et al., 2013). It has 10,299 windows from 30 volunteers, 561 hand-crafted time/frequency features per window, and 6 classes (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING).
- **Task:** 6-class supervised classification of the activity from one 2.56-second sensor window.
- **Main metric:** **macro-F1** on the official test set, which has 9 held-out subjects. Accuracy is reported too, but macro-F1 weights every activity equally and exposes weak classes.
- **Why this dataset:** It is small enough to train from-scratch NumPy models in seconds, it is clean (no missing values), it has a meaningful difficulty (SITTING vs STANDING), and it is a subject-wise split, so it tests generalization to *new people*. I will keep this single dataset for the whole semester so that models can be compared directly.
- **Evaluation protocol:** The training subjects are split *by subject* into train (80%) and validation (20%), which avoids leakage between correlated windows from the same person. Standardization statistics come from the training part only. The test set is used once per final model.

## 2. What was covered in lectures

| Week | Topic | Key ideas I need to remember |
|---|---|---|
| W01 | Introduction to ML | Mitchell's $T$/$P$/$E$ definition; supervised, unsupervised and reinforcement learning; data → features → model → training → evaluation → deployment; precision, recall, F1, confusion matrix; under/over-fitting; bias–variance decomposition $\mathbb{E}\left[(y - \hat{y})^2\right] = \text{bias}^2 + \text{variance} + \text{noise}$; the i.i.d. assumption for train/val/test. |
| W02 | Decision Tree | Non-parametric flowchart model; internal nodes (feature tests), edges (outcomes), leaves (majority class or target mean); impurity metrics: Entropy $H(S) = -\sum p_i \log_2 p_i$, Gini $G(S) = 1 - \sum p_i^2$, and variance/SSE reduction; ID3: multiway splits via maximum Information Gain $IG(S,A) = H(S) - \sum \frac{\|S_v\|}{\|S\|} H(S_v)$ (favors many-valued attributes); C4.5: uses Gain Ratio $GR = \frac{IG}{\text{SplitInfo}}$ to penalize wide splits, handles continuous splits by thresholding midpoints, fractional missing value weighting, and pessimistic error post-pruning; CART: strictly binary splits, uses Gini (classification) or SSE (regression), handles missing values via surrogate splits, and avoids overfitting via cost-complexity pruning $R_\alpha(T) = R(T) + \alpha\|T\|$ with cross-validation. |
| W03 | Linear regression & Linear classification | Gaussian noise model $t = w^\top x + \varepsilon$; maximum likelihood gives the sum-of-squares error; normal equation $w = (X^\top X)^{-1} X^\top t$; noise precision $\beta$; mini-batch gradient descent $\Delta w = \dfrac{1}{B} \sum (w^\top x_n - t_n)x_n$; basis functions for nonlinear relations; overfitting with polynomial degree $M$; L2 (ridge) regularization; MSE/RMSE. <br /> Perceptron and delta rule; logistic regression $\hat{y} = \sigma(w^\top x)$; the Bernoulli likelihood gives binary cross-entropy; gradient $X^\top(\hat{y} - y)$; Hessian $X^\top R X$ (Newton/IRLS); softmax regression with categorical cross-entropy; linear decision boundaries. |
| W04 | MLP and training ANNs | FC layers and activations (sigmoid, tanh, ReLU, Leaky ReLU, SiLU); MLP as a composition of affine and nonlinear maps; forward/backward/update; losses (MSE, MAE, Huber, BCE, CE, focal); optimizers (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW); LR schedules, L2, dropout, normalization, early stopping; He/Xavier init; vanishing/exploding gradients. |

## 3. What I built during the catch-up

- **Data pipeline** ([src/data.py](../../src/data.py)): loads the raw text files, maps labels to 0–5, does a subject-wise validation split, and standardizes using training statistics only.
- **Metrics from scratch** ([src/metrics.py](../../src/metrics.py)): accuracy, confusion matrix, per-class precision/recall/F1 and their macro average. The unit test checks them against scikit-learn.
- **W03 models** ([src/from_scratch/logistic.py](../../src/from_scratch/logistic.py)): perceptron, delta rule (ADALINE), binary logistic regression and softmax regression, all trained with (mini-batch) gradient descent.
- **W04 model** ([src/from_scratch/mlp.py](../../src/from_scratch/mlp.py)): an MLP with an explicit forward pass, backpropagation, SGD/Momentum/Adam and early stopping. The backward pass is verified with a finite-difference gradient check for three activations ([tests/test_gradients.py](../../tests/test_gradients.py)).
- **Reference adapters** ([src/reference_adapters/sklearn_ref.py](../../src/reference_adapters/sklearn_ref.py)): scikit-learn models used *only* as sanity checks.

## 4. Results so far (official test set, macro-F1)

| Model | Setting | Test macro-F1 |
|---|---|---|
| Softmax regression (mine) | $\eta = 0.1$, 30 epochs, no L2 | 0.9365 |
| sklearn LogisticRegression (reference) | L2, $C = 1$ | 0.9556 |
| MLP 561-128-6 (mine) | ReLU, Adam $\eta = 10^{-3}$, early stopping | **0.9465** |
| sklearn MLPClassifier (reference) | ReLU, Adam | 0.9428 |

All runs are logged in [results/metrics.csv](../../results/metrics.csv) and [MODEL_LOG.md](../../MODEL_LOG.md).

## 5. Open questions and gaps

- SITTING vs STANDING is the dominant error for every model so far. I want to find out whether it is a feature problem (the postures are almost identical for the accelerometer) or a model problem.
- My softmax regression is about 2 F1 points behind the regularized sklearn model, so the next step is to add and tune L2.
- My linear regression (W02) was studied only on paper. HAR has no continuous target, so I kept W02 theory-only. The normal equation and GD derivations are in the A4 notes draft.
- I have not yet tried learning-rate schedules, dropout or deeper MLPs.

## 6. Plan from W05

For each week: a handwritten drill (first attempt, then corrections), the model coded from scratch and applied to the same HAR split, one controlled experiment, then a blog, a PROGRESS row and a week tag. Decision trees are next (W05). A tree gives a very different, non-linear, axis-aligned model to compare against the linear and MLP baselines above, using the same macro-F1 protocol.
