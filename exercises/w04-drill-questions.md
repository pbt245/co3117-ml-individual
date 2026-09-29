# W04 Drill - MLP and Backpropagation (15 min, closed book)

Write the first attempt on paper and scan it to `exercises/w04-first-attempt.pdf`, then correct it in [w04-corrections.md](w04-corrections.md).

1. Why does an MLP need a nonlinear activation? Show what a 2-layer network without activation collapses to.
2. Write the forward pass of a 1-hidden-layer MLP with a softmax output, and the backward-pass formulas for $\delta$ at the output layer, $\delta$ at the hidden layer, and $\dfrac{\partial L}{\partial W}$ for both layers.
3. Compute by hand for a 1-1-1 network with sigmoid units and BCE: $x = 1$, $w_1 = 0.5$, $w_2 = 1$, biases 0, $y = 1$. Give $\hat{y}$, $L$, $\dfrac{\partial L}{\partial w_2}$ and $\dfrac{\partial L}{\partial w_1}$.
4. Write the update rules of SGD with momentum and of Adam. What is Adam's bias correction for, and why does early stopping act as regularization?
