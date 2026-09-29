# Release Baseline Questions (w01 and w02)

Answer on paper without slides, then scan the result to `exercises/release-baseline-w01-w02.pdf`.
This is a diagnostic of *where I stand*. Do not correct the scan afterwards; write any corrections in a separate file.

1. **(W01)** State Mitchell's definition of machine learning. Identify $T$, $P$ and $E$ for this repo's use case (recognising activities from smartphone sensors).
2. **(W01)** Give one example each of supervised, unsupervised and reinforcement learning. For each, say what the "label" or "signal" is.
3. **(W01)** A classifier on 100 samples gives $TP = 30$, $FP = 10$, $FN = 20$, $TN = 40$ for the positive class. Compute accuracy, precision, recall and F1. Explain why accuracy alone can be misleading.
4. **(W01)** Draw the typical training-error and validation-error curves against model complexity. Mark the underfitting, good-fit and overfitting regions.
5. **(W01)** Write the bias–variance decomposition of the expected squared error, and explain in one sentence what each term means.
6. **(W02)** A node contains 9 *Yes* and 5 *No* samples. Compute its entropy $H(S)$ and its Gini index. What are the minimum and maximum values of each measure for 2 classes, and when are they reached?
7. **(W02)** In Play Tennis, splitting on *Wind* gives Weak = 6 Yes / 2 No and Strong = 3 Yes / 3 No. Compute the information gain $IG(S, \text{Wind})$. Given $IG(\text{Outlook}) = 0.25$, which attribute does ID3 put at the root, and why is this choice called *greedy*?
8. **(W02)** Why does information gain favour attributes with many values (e.g. a unique *Day* ID)? Write the gain ratio used by C4.5 and compute $\mathrm{SplitInfo}$ for an attribute that puts each of the 14 samples in its own branch.
9. **(W02)** How does a tree split a *continuous* attribute such as Temperature? A fully grown tree has training error 0 but a high test error. Explain why, then compare pre-pruning and post-pruning, and write the CART cost-complexity objective $R_\alpha(T)$.
