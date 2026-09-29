# W04 - Multilayer Perceptron and Backpropagation

Slides: `MLP.pdf`, `TrainingANN.pdf`. Code: [mlp.py](../../src/from_scratch/mlp.py), [w04_mlp.py](../../experiments/part1_pre_midterm/w04_mlp.py)

## A. Concept capsule
- **Model:** A stack of [FC + activation] layers, $h^{(l)} = \varphi\left(W^{(l)} h^{(l-1)} + b^{(l)}\right)$, followed by a linear softmax head. The hidden layers learn nonlinear features, and the head is softmax regression on those features.
- **Key assumptions:** Samples are i.i.d. and $\varphi$ is nonlinear; without $\varphi$, stacked FC layers collapse into a single linear map.
- **Objective / update rule:** Minimise the cross-entropy. Each iteration has three steps:
  - Forward pass: compute and cache every $z$ and $h$.
  - Backward pass: $\delta^{L} = P - Y$ and $\delta^{l} = \left(\delta^{l+1} W^{(l+1)}\right) \odot \varphi'\left(z^{(l)}\right)$, then $\dfrac{\partial L}{\partial W^{(l)}} = \delta^{l\top} h^{(l-1)}$.
  - Update step: SGD, Momentum or Adam.

## B. Worked example
A 1-1-1 network with sigmoid units and BCE. Values: $x = 1$, $w_1 = 0.5$, $w_2 = 1$, biases 0, $y = 1$, $\eta = 0.5$.
- Forward: $h = \sigma(0.5) = 0.6225$, $\hat{y} = \sigma(0.6225) = 0.6508$, so $L = 0.4296$.
- Backward, output layer: $\delta_2 = \hat{y} - y = -0.3492$, so $\dfrac{\partial L}{\partial w_2} = \delta_2 h = -0.2174$.
- Backward, hidden layer: $\delta_1 = \delta_2 w_2 h(1 - h) = -0.0821$, so $\dfrac{\partial L}{\partial w_1} = -0.0821$.
- Update: $w_2 = 1.1087$, $b_2 = 0.1746$, $w_1 = 0.5410$, $b_1 = 0.0410$. The new loss is **0.3453**.

## C. Code-to-theory trace
- **Forward pass $z = hW^\top + b$, $h = \varphi(z)$:** `MLP.forward()`, lines 46–51.
- **Output error $\delta = \dfrac{P - Y}{B}$** (softmax + CE combined): `MLP.backward()`, line 61.
- **Gradients $\dfrac{\partial L}{\partial W} = \delta^\top h_{\text{prev}} + \lambda W$ and $\dfrac{\partial L}{\partial b} = \sum \delta$:** lines 63–64.
- **Backpropagating $\delta$ through $W$ and $\varphi'(z)$:** line 67.
- **Optimizers:** SGD at line 75. Momentum at lines 77–78. Adam with bias correction at lines 80–84.
- **SGD training loop and early stopping:** `MLP.fit()`, lines 95–116. The best checkpoint is saved at line 109 and restored at lines 114–116.
- **Backprop verification:** a finite-difference check in [test_gradients.py](../../tests/test_gradients.py) gives relative error below $10^{-5}$.

## D. Controlled experiment
- **Question:** How does the optimizer affect training of the same MLP (561-128-6, ReLU, 40 epochs, batch size 64)?
- **Setup:** SGD with $\eta = 0.01$, Momentum with $\eta = 0.01$ and $\mu = 0.9$, and Adam with $\eta = 0.001$. The seed, init and split are fixed.
- **Result:** Validation macro-F1 was **0.921 / 0.935 / 0.943**. Adam hits its minimum validation loss (0.160) at epoch 6, while SGD needs 35 epochs to reach 0.195. After epoch 6, Adam's training loss drops to 0.0003 but its validation loss rises to 0.199, which is overfitting.
- **Conclusion:** Momentum and Adam converge faster but overfit sooner. The final model uses Adam + early stopping (patience 5) and stops at epoch 11, reaching **test macro-F1 = 0.9465**. The sklearn MLP gets 0.9428 and my W03 softmax got 0.9365.

## E. Failure / Misconception
- **Observed error:** SITTING/STANDING are still the weakest classes (F1 0.917 and 0.924), although both improve relative to softmax regression.
- **Cause:** The two postures are nearly identical in the 561 summary features, so the remaining error comes more from the data and features than from model capacity.
- **Common misconception:** Computing the output error as $(P - Y) \odot \mathrm{softmax}'(z)$. For softmax combined with CE the Jacobian cancels, so $\delta = P - Y$. A gradient check catches this bug.

## F. Written-exam capsule
A multilayer perceptron composes affine transformations with nonlinear activations, so its hidden layers learn a representation in which a linear softmax head can separate the classes. Training minimises a loss such as cross-entropy by gradient descent. Each iteration does a forward pass that stores activations, a backward pass that applies the chain rule layer by layer to get every gradient in one sweep (backpropagation), and an optimizer update such as SGD, Momentum or Adam. A hidden layer's error is the next layer's error times the transposed weights times $\varphi'$, so saturating activations such as sigmoid can make gradients vanish. The high capacity of an MLP calls for regularization, for example early stopping.

## G. Reflection
- **Understood:** Why the forward pass caches $z$ and $h$: backward needs $h^{(l-1)}$ for $\partial W$ and $z^{(l)}$ for $\varphi'$.
- **Still unclear / next:** Tuning L2 and hidden size together, and whether a second hidden layer helps on HAR.

## H. Inquiry trail
<!-- To be filled in by me (AI use disclosure). -->
