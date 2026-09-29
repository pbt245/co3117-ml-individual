# CO3117 Machine Learning — Individual Assignment

A single use case followed for the whole semester: **6-class human activity recognition** on the
[UCI HAR dataset](data/README.md). Each week's model is implemented **from scratch in NumPy**, evaluated on the
same subject-wise protocol, and compared against a scikit-learn reference.

- **Main metric:** macro-F1 on the official test set (9 unseen subjects).
- **Validation:** 20% of the *training subjects* are held out, so the split is by subject and does not leak.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# download the dataset into data/raw/ (see data/README.md)
.venv/bin/pytest                                              # gradient checks + metric tests
.venv/bin/python -m experiments.part1_pre_midterm.w03_logreg  # W03: perceptron / delta / logistic / softmax
.venv/bin/python -m experiments.part1_pre_midterm.w04_mlp     # W04: MLP + backprop
```

Each experiment prints its results, appends rows to `results/metrics.csv` (its own week's old rows are replaced),
and writes figures to `results/figures/`.

## Results so far (test macro-F1)

| Week | Model (from scratch) | Test macro-F1 | sklearn reference |
|---|---|---|---|
| W02 | Decision Tree | TBD | TBD |
| W03 | Softmax regression ($\eta = 0.1$) | 0.9365 | 0.9556 |
| W04 | MLP 561-128-6, ReLU, Adam, early stopping | 0.9465 | 0.9428 |

## Repository map

| Path | Content |
|---|---|
| [PROGRESS.md](PROGRESS.md) | Weekly dashboard (blog, drill, commits, tags) |
| [MODEL_LOG.md](MODEL_LOG.md) | Every trained model with its setting and scores |
| [AI_USE.md](AI_USE.md) | AI-use disclosure |
| [REFERENCES.md](REFERENCES.md) | Slides, dataset citation, library docs |
| [docs/](docs/index.md) | Pre-release catch-up and weekly blogs |
| [src/](src/) | `data.py`, `metrics.py`, `from_scratch/` models, `reference_adapters/` (sklearn) |
| [experiments/](experiments/) | One script per week (Part I / Part II) |
| [tests/](tests/) | Finite-difference gradient checks, metric tests |
| [results/](results/) | `metrics.csv` and figures |
| [exercises/](exercises/) | Handwritten drills (scans) and corrections |
| [exam/](exam/) | A4 notes, mock exams, reflections |
| [report/](report/) | Part I summary and Part II final report |
