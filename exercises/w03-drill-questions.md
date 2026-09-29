# W03 Drill - Perceptron, Delta Rule, Logistic Regression (15 min, closed book)

Write the first attempt on paper and scan it to `exercises/w03-first-attempt.pdf`, then correct it in [w03-corrections.md](w03-corrections.md).

1. Write the perceptron update rule and the delta (Widrow–Hoff) rule. State two differences between them: the output they use and the loss they minimise.
2. Starting from $p(y \mid x, w) = \hat{y}^{y}(1 - \hat{y})^{1 - y}$ with $\hat{y} = \sigma(w^\top x)$, derive the negative log-likelihood of $N$ i.i.d. samples. Name the resulting loss.
3. Using $\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)$ and the chain rule, show that $\dfrac{\partial L}{\partial w} = (\hat{y} - y)x$ for one sample.
4. Do one gradient step by hand: $x = [1, 1, 2]$ (bias first), $y = 1$, $w = 0$, $\eta = 0.1$. Give the new $w$ and check that the loss decreases.
