# W03 — Perceptron, Delta Rule and Logistic/Softmax Regression

Slides: `LinearRegression.pdf` (the delta rule reuses its GD update), `LogisticRegression.pdf`, recap in `MLP.pdf`. Code: [logistic.py](../../src/from_scratch/logistic.py), [w03_logreg.py](../../experiments/part1_pre_midterm/w03_logreg.py)

## A. Concept capsule
- **Model:** A linear score $z = w^\top x$ (bias absorbed through $x = [1, x]$) followed by an output function: a step (perceptron), the identity (delta rule), a sigmoid (logistic) or a softmax (K classes).
- **Key assumptions:** Samples are i.i.d. and the classes are (nearly) linearly separable, since the boundary $w^\top x = 0$ is a hyperplane. The label is $\mathrm{Bernoulli}(\hat{y})$.
- **Objective / update rule:**
  - Perceptron: $w \leftarrow w + \eta(y - \hat{y})x$, applied only on mistakes.
  - Delta rule: $w \leftarrow w - \dfrac{\eta}{B} X^\top(Xw - t)$.
  - Logistic: the negative log-likelihood (NLL) is the BCE, with $\nabla L = X^\top(\hat{y} - y)$ (eq. 3.9).
  - Softmax: $\nabla L = X^\top(P - Y)$.

## B. Worked example
One GD step with $x = [1, 1, 2]$, $y = 1$, $w = 0$, $\eta = 0.1$:
- $\hat{y} = \sigma(0) = 0.5$, so $L = 0.693$
- $\nabla L = (\hat{y} - y)x = [-0.5, -0.5, -1]$, so $w = [0.05, 0.05, 0.10]$
- Now $z = 0.3$, $\hat{y} = 0.574$ and $L = 0.554$. The loss decreased.
- Perceptron on the same sample with $y = 0$: it predicts $\mathrm{step}(0) = 1$, which is wrong, so $w = -0.1 \cdot x = [-0.1, -0.1, -0.2]$.

## C. Code-to-theory trace
- **Stable sigmoid:** `sigmoid()`, lines 12–19.
- **Stable softmax:** `softmax()`, lines 22–25. Line 23 subtracts the row max, which is allowed because softmax is shift-invariant.
- **Perceptron rule:** `Perceptron.fit()`, line 63, inside the per-sample loop at lines 58–65.
- **Delta rule:** `DeltaRule.fit()`, line 89.
- **Logistic gradient $X^\top(\hat{y} - y)$:** `LogisticRegression.fit()`, line 113.
- **Softmax gradient $\dfrac{X^\top(P - Y)}{B} + \lambda W$:** `SoftmaxRegression.gradient()`, lines 142–143. The GD loop is at lines 152–157.

## D. Controlled experiment
- **Question:** How does the learning rate $\eta$ affect softmax regression (6 classes, 30 epochs, batch size 64)?
- **Setup:** $\eta \in \lbrace 0.001, 0.01, 0.1, 1.0 \rbrace$, with the same seed and the same subject-wise train/val split.
- **Result:** Validation macro-F1 was 0.870 → 0.910 → **0.925** → 0.910. With $\eta = 1.0$ the validation loss reaches 0.758, against 0.183 for $\eta = 0.1$. The best model reaches **test macro-F1 = 0.9365**. The sklearn reference (L2, $C = 1$) reaches 0.9556.
- **Conclusion:** A small $\eta$ underfits within the epoch budget, and a large $\eta$ overshoots into overconfident weights. The gap to sklearn points to adding L2 next.

## E. Failure / Misconception
- **Observed error:** SITTING and STANDING are the worst classes (test F1 0.888 and 0.902, against 0.991 for LAYING). On SITTING-vs-STANDING the perceptron never reaches zero mistakes (310 → about 100 per epoch), while on WALKING-vs-LAYING it reaches zero at epoch 2.
- **Cause:** This violates the **linear separability** assumption of the perceptron convergence theorem. The two postures have almost identical features, so the perceptron oscillates. Logistic regression still converges to the maximum-likelihood boundary (test F1 0.926).
- **Common misconception:** Treating the delta rule and the perceptron as the same thing. The delta rule minimises squared error on the *linear output*, while the perceptron uses the thresholded output and only moves on mistakes.

## F. Written-exam capsule
Logistic regression models the probability of the positive class as the sigmoid of a linear score, $\hat{y} = \sigma(w^\top x)$. Its weights are found by maximum likelihood: with i.i.d. Bernoulli labels, the negative log-likelihood is the binary cross-entropy. That loss is convex but has no closed-form minimiser, so it is minimised iteratively with gradient descent (gradient $X^\top(\hat{y} - y)$) or Newton's method (Hessian $X^\top R X$). The decision boundary $w^\top x = 0$ is a hyperplane, so classes that are not linearly separable cannot be split perfectly. Softmax regression extends this to $K$ classes, and the perceptron and delta rule differ only in the output function and the error they minimise.

## G. Reflection
- **Understood:** Why the delta rule, logistic regression and softmax regression all share the gradient form "error × input", $(\hat{y} - y)x$.
- **Still unclear / next:** Tuning L2, and implementing the Newton/IRLS update.

## H. Inquiry trail
<!-- To be filled in by me (AI use disclosure). -->
