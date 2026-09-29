# A4 Notes draft, W01–W04

## W01 — Basics and evaluation
- **Mitchell's definition:** a program learns from experience $E$ on task $T$, measured by $P$, if $P$ on $T$ improves with $E$.
- **Learning types:** supervised ($x \to y$), unsupervised ($x$ only: clustering, PCA, density), reinforcement (state, action, reward; maximise cumulative reward).
- **Confusion-matrix metrics:**
  - $\mathrm{Acc} = \dfrac{TP + TN}{N}$
  - $\mathrm{Prec} = \dfrac{TP}{TP + FP}$
  - $\mathrm{Rec} = \dfrac{TP}{TP + FN}$
  - $F_1 = \dfrac{2PR}{P + R}$
  - Macro-F1 is the mean of the per-class F1 scores. Every class weighs the same, which exposes weak classes.
- **Fit:** underfit means both errors are high (high bias). Overfit means train error is low but val error is high (high variance).
- **Bias–variance decomposition:** $\mathbb{E}\left[(y - \hat{y})^2\right] = \mathrm{Bias}[\hat{y}]^2 + \mathrm{Var}[\hat{y}] + \sigma^2$ ($\sigma^2$ is the irreducible noise).
- **Data splits:** train / val / test are i.i.d. The test set is touched once. Preprocessing is fitted on train only.

## W02 — Decision trees
- **Tree:** internal node = test on one feature ($x_j \le T$ or $A = v$), leaf = majority class (classification) or mean target (regression). It is grown greedily top-down; prediction walks root → leaf.
- **Entropy:** $H(S) = -\sum_i p_i \log_2 p_i$. It is 0 when the node is pure and 1 for a 50/50 split of 2 classes. Play Tennis (9 Yes / 5 No): $H = 0.940$.
- **Information gain (ID3):** $IG(S, A) = H(S) - \sum_{v} \dfrac{|S_v|}{|S|} H(S_v)$. Pick $\arg\max IG$. Play Tennis: Outlook 0.247 > Humidity 0.152 > Wind 0.048 > Temp 0.029, so the root is Outlook.
- **Gain ratio (C4.5):** $GR = \dfrac{IG}{\mathrm{SplitInfo}}$ with $\mathrm{SplitInfo}(S, A) = -\sum_v \dfrac{|S_v|}{|S|} \log_2 \dfrac{|S_v|}{|S|}$. It fixes IG's bias toward many-valued attributes. Outlook: $0.247 / 1.577 = 0.156$.
- **Gini (CART):** $G(t) = 1 - \sum_k p(k \mid t)^2$, with a maximum of 0.5 for 2 classes. Play Tennis: $0.459$. Split quality: $\Delta G = G(t) - \dfrac{N_L}{N_t} G(t_L) - \dfrac{N_R}{N_t} G(t_R)$.
- **Regression tree (CART):** $I(t) = \sum_{i \in t} (y_i - \bar{y}_t)^2$, maximise $\Delta I = I(t) - I(t_L) - I(t_R)$. The leaf predicts $\bar{y}$.
- **Continuous attribute:** sort the values, try the midpoints between consecutive distinct values, and keep the best $A \le T^*$.
- **Pre-pruning:** stop when the node is small ($|S| < N_{min}$), the gain is tiny, the node is nearly pure, or the depth reaches its maximum.
- **Post-pruning:** replace a subtree by a leaf if the leaf error is not worse. C4.5 uses a pessimistic error $\hat{p} + z\sqrt{\hat{p}(1 - \hat{p})/N}$. The reduced-error variant uses a validation set.
- **Cost-complexity (CART):** $R_\alpha(T) = R(T) + \alpha |T|$, where $|T|$ = number of leaves. Weakest link: $g(t) = \dfrac{R(t) - R(T_t)}{|T_t| - 1}$. Choose $\alpha$ by cross-validation (optionally with the 1-SE rule).
- **Missing values:** C4.5 sends a sample fractionally down all branches; CART uses surrogate splits; ID3 needs imputation.
- **Comparison:** ID3 = multiway splits + IG, categorical only. C4.5 = gain ratio, continuous attributes, pruning. CART = binary splits + Gini/SSE, cost-complexity pruning.

## W03 — Linear models: regression and classification
### Linear regression
- **Model:** $t = w^\top x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \beta^{-1})$, so $p(t \mid x) = \mathcal{N}(w^\top x, \beta^{-1})$.
- **NLL:** $L = \beta E_D(w) - \dfrac{N}{2} \ln \beta + \text{const}$, with $E_D = \dfrac{1}{2} \sum (t_n - w^\top x_n)^2$.
- **Normal equation:** $w_{ML} = (X^\top X)^{-1} X^\top t$, with $X$ of size $N \times D$ and $X^\top X$ of size $D \times D$. The noise estimate is $\dfrac{1}{\beta_{ML}} = \dfrac{1}{N} \sum (t_n - w_{ML}^\top x_n)^2$.
- **Gradient descent:** $\Delta w = \dfrac{1}{B} \sum (w^\top x_n - t_n)x_n$, then $w \leftarrow w - \eta \Delta w$. Stop when $\lvert e_t - e_{t-1} \rvert$ is small.
- **Basis functions:** $x \to [1, \phi_1(x), \dots, \phi_{M-1}(x)]$. The model is nonlinear in $x$ but still linear in $w$, so the same solution applies.
- **Ridge:** $\dfrac{1}{2} \sum (t - w^\top x)^2 + \dfrac{\lambda}{2} \lVert w \rVert^2$, giving $w = (\lambda I + X^\top X)^{-1} X^\top t$.
- **Metrics:** $\mathrm{MSE} = \dfrac{1}{N} e^\top e$ and $\mathrm{RMSE} = \sqrt{\mathrm{MSE}}$, which is in the same unit as $t$.

### Linear classification
- **Perceptron:** $\hat{y} = \mathrm{step}(w^\top x)$, $w \leftarrow w + \eta(y - \hat{y})x$. It updates only on mistakes and converges only if the data are linearly separable.
- **Delta rule (ADALINE):** the output is $o = w^\top x$, and the update is $w \leftarrow w - \eta(o - t)x$.
- **Logistic regression:** $\hat{y} = \sigma(w^\top x)$, with $\sigma' = \sigma(1 - \sigma)$.
  - Likelihood: $p(y \mid x) = \hat{y}^{y}(1 - \hat{y})^{1 - y}$.
  - Loss (BCE): $L = -\sum \left[y \ln \hat{y} + (1 - y) \ln(1 - \hat{y})\right]$.
- **Gradient:** $\nabla L = X^\top(\hat{y} - y)$. **Hessian:** $H = X^\top R X$ with $R = \mathrm{diag}\big(\hat{y}_n(1 - \hat{y}_n)\big)$, used by Newton/IRLS: $w \leftarrow w - H^{-1} \nabla L$.
- **Softmax regression:** $p_k = \dfrac{e^{z_k}}{\sum_j e^{z_j}}$, with loss $\mathrm{CE} = -\sum_k y_k \ln p_k$ and gradient $X^\top(P - Y)$. The boundary between classes $k$ and $j$ is $(w_k - w_j)^\top x + (b_k - b_j) = 0$.
- **Numerical stability:** subtract $\max z$ before the softmax. The stable BCE-with-logits is $\max(z, 0) - yz + \ln\left(1 + e^{-\lvert z \rvert}\right)$.

## W04 — MLP and training
- **Layers:** $h^{(l)} = \varphi\left(W^{(l)} h^{(l-1)} + b^{(l)}\right)$. An FC layer with $N$ inputs and $M$ outputs has $M \cdot N + M$ parameters.
- **Activations:**
  - sigmoid: $\sigma' = \sigma(1 - \sigma)$
  - tanh: $\tanh' = 1 - \tanh^2$
  - ReLU: derivative $\mathbb{1}[z > 0]$
  - Leaky ReLU: slope $\alpha$ for $z < 0$
  - SiLU: $z\sigma(z)$
- **Backprop:**
  - Output: $\delta^{L} = P - Y$ (softmax + CE)
  - Hidden: $\delta^{l} = \left(W^{(l+1)\top} \delta^{l+1}\right) \odot \varphi'\left(z^{(l)}\right)$
  - Gradients: $\dfrac{\partial L}{\partial W^{(l)}} = \delta^{l} h^{(l-1)\top}$, $\dfrac{\partial L}{\partial b^{(l)}} = \delta^{l}$
- **Losses:**
  - MSE, and MAE (robust to outliers)
  - Huber: $\dfrac{1}{2} e^2$ if $\lvert e \rvert \le \delta$, otherwise $\delta\left(\lvert e \rvert - \dfrac{1}{2}\delta\right)$
  - Focal: $-\alpha(1 - p)^{\gamma} \ln p$
- **Optimizers:**
  - SGD: $\theta \leftarrow \theta - \eta g$
  - Momentum: $v = \mu v + g$, $\theta \leftarrow \theta - \eta v$
  - NAG: evaluates $g$ at $\theta - \eta \mu v$
  - AdaGrad: $G \mathrel{+}= g^2$, step $\dfrac{\eta g}{\sqrt{G + \varepsilon}}$
  - RMSProp: $s = \rho s + (1 - \rho)g^2$
  - Adam: $m = \beta_1 m + (1 - \beta_1)g$, $v = \beta_2 v + (1 - \beta_2)g^2$, $\hat{m} = \dfrac{m}{1 - \beta_1^t}$, $\hat{v} = \dfrac{v}{1 - \beta_2^t}$, $\theta \leftarrow \theta - \eta \dfrac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}$. Defaults: $\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.
  - AdamW: $\theta \leftarrow (1 - \eta\lambda)\theta - \eta \dfrac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}$
- **Regularization:** L2 $\dfrac{\lambda}{2} \lVert \theta \rVert^2$, dropout ($p = 0.1$–$0.5$), BatchNorm/LayerNorm, and early stopping (patience, keep the best checkpoint).
- **Initialization:** Xavier for sigmoid/tanh, He for ReLU.
- **Gradient problems:** vanishing (fix with ReLU, residual connections, normalization) and exploding (fix with clipping).
- **Vocabulary:** epoch = one full pass, iteration = one mini-batch update. With a large $\eta$ training diverges; with a small $\eta$ it is slow.
