*English · [Русский](README.ru.md)*

# Spotify Churn Analysis

A churn prediction project that ends with a negative result — and a set of experiments
demonstrating why that result is correct rather than a modelling failure.

## Overview

The dataset contains no learnable relationship between its features and the churn label.
Cross-validated ROC-AUC sits at **0.50** for every model and every preprocessing strategy
tried, which is the score of random guessing.

An initial gradient boosting model reached F1 = 0.193, and the obvious next step would have
been to tune it — balance the classes, engineer features, apply PCA. Instead of assuming the
model was at fault, this repository tests that assumption directly, through five experiments:
a sanity check of the pipeline on data with known signal, a comparison of the real labels
against deliberately destroyed ones, a breakdown of what class balancing does to the metrics,
a calibration of the mutual information estimator, and a demonstration of overfitting. All
five converge on the same answer: nothing in the data supports prediction, so no amount of
preprocessing can help.

The value of the project is therefore methodological. It shows how to distinguish *"the model
needs work"* from *"the data has nothing in it"* before spending effort on the wrong problem.

## Dataset

[Spotify User Behavior for Churn Analysis](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis/data) (Kaggle) — 8,000 users, 12 columns.

| Group | Columns |
|---|---|
| Demographics | `gender`, `age`, `country` |
| Subscription | `subscription_type` |
| Listening behaviour | `listening_time`, `songs_played_per_day`, `skip_rate`, `ads_listened_per_week`, `offline_listening` |
| Technical | `device_type` |
| Target | `is_churned` (1 = churned, 0 = retained) |

Churn rate is 25.89%, giving a class ratio of about 2.9 : 1 — an imbalance that is moderate
and well within the range models handle routinely.

## Key concepts

Three terms appear throughout the experiments, so they are defined once here.

**Model score.** A classifier does not output a ready answer of "churned / retained". Its
`predict_proba` method assigns each user a number between 0 and 1 — an estimate of how likely
churn is. Turning scores into answers requires a **threshold**: a rule such as "everyone
scoring above 0.5 counts as churned". The threshold value is an arbitrary choice, and every
threshold-based metric depends on it.

**ROC-AUC.** A ranking-quality metric that does not depend on the threshold. The definition
is literal: it is the fraction of (churned, retained) pairs in which the churned user
received the higher score. A value of 1.0 means every churner ranks above every retained
user; a value of **0.5 means the ordering of scores is random and the model is equivalent to
a coin flip**. Because it ignores the threshold, ROC-AUC shows whether the model separates
the classes at all, which is what makes it the primary diagnostic metric.

**Held-out set.** The data is split into a training part and a test part; the model is fitted
on the first and evaluated exclusively on the second, which it has never seen. The
experiments use either a 70/30 split or cross-validation — the same scheme repeated five
times with a rotating held-out part and the results averaged. Why this is mandatory is shown
concretely by experiment 5.

## Method

Each experiment answers one question, and they build on each other. The sections share a
single structure: what is tested, how, the result, the conclusion.

### 1. Is the pipeline working?

**What is tested.** Before concluding that the data is empty, a bug in the code has to be
ruled out: broken cross-validation or a miscomputed metric would also produce 0.50 on any
data. From the outside the two situations look identical, so the instrument is tested
separately — on data where signal is known to exist.

**How.** An artificial column equal to `target + random noise` is added to a copy of the
data — related to the answer by construction. The gap between the classes in this column is
always exactly 1.0 (churners get +1, retained users +0), while the spread of the noise is set
by hand. The ratio of gap to spread is the signal strength: at a spread of 0.8 the classes
are nearly separated, at 6.0 they are almost fully mixed. Training and evaluation run through
the same code as every other experiment.

**Result.**

| Injected signal strength (noise spread) | ROC-AUC |
|---|---|
| very weak (6.0) | 0.5155 |
| weak (3.0) | 0.5750 |
| moderate (1.5) | 0.6569 |
| strong (0.8) | 0.8097 |

**Conclusion.** The metric rises monotonically with signal strength and reacts even to the
very weak case, where the noise is six times the gap between classes. The pipeline is sound
and sensitive; null results in the following experiments describe the data, not a bug.

### 2. Real target vs shuffled target

**What is tested.** Whether any of the planned improvements — dropping `country`, PCA,
feature engineering, class balancing — extracts anything from the data at all.

**How.** Shuffling `is_churned` across rows destroys any relationship by construction: each
label lands on a random user. Every method is run twice — once on the real labels, once on
the shuffled ones. The shuffled version acts as a control group: a method that scores the
same on it as on the real labels is extracting nothing.

**Result.**

| Method | ROC-AUC (real target) | ROC-AUC (shuffled target) |
|---|---|---|
| baseline | 0.5040 | 0.4839 |
| drop `country` | 0.5125 | 0.4920 |
| PCA (8 components) | 0.5055 | 0.4973 |
| feature engineering (+7 features) | 0.5078 | 0.4942 |
| SMOTE (class balancing) | 0.4921 | 0.4837 |

![Real vs shuffled target](reports/figures/real_vs_shuffled.png)

**Conclusion.** The columns are indistinguishable: the data behaves exactly like data whose
labels were deliberately destroyed. The `country` case is instructive: dropping it raised
ROC-AUC by 0.009 — but the same move raised the *shuffled* version by 0.008. That is
cross-validation noise, not a finding. Picking the best of several such variants is how a
project talks itself into a result that is not there.

### 3. What class balancing actually does

**What is tested.** The initial model scored F1 = 0.193, and the improvement plan proposed
raising it through class balancing. The experiment checks whether balancing improves the
model — or merely moves a point along a trade-off scale.

Three threshold-based metrics are needed here. **Precision** is the fraction of true
churners among the users the model flagged as churning. **Recall** is the fraction the model
caught among all actual churners. **F1** is their harmonic mean, folding both into a single
number. All three depend on the threshold: changing the cut-off rule changes them without
any change to the model.

**How.** One and the same model is evaluated five ways: as is (threshold 0.5), after SMOTE
balancing, at thresholds 0.30 and 0.20, and next to a constant "everyone churns" predictor —
the floor any useful model must beat. ROC-AUC is reported in every row as the control: if the
model genuinely improved it would rise; if only the threshold moved, it stays put.

**Result.**

| Variant | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| no balancing | 0.5056 | 0.5000 | 0.0032 | 0.0064 |
| SMOTE | 0.4964 | 0.1864 | 0.0177 | 0.0324 |
| threshold 0.30 | 0.5056 | 0.2585 | 0.1948 | 0.2222 |
| threshold 0.20 | 0.5056 | 0.2625 | 0.9130 | 0.4078 |
| always predict churn | 0.5000 | 0.2587 | 1.0000 | 0.4111 |

![F1 vs threshold](reports/figures/f1_vs_threshold.png)

**Conclusion.** Lowering the threshold alone pushes F1 from 0.006 to 0.408 — a seemingly
dramatic improvement, until the last row is read: the constant predictor scores 0.411 and
still wins. AUC meanwhile never moves in any row, meaning the model itself never improved
once.

Precision never leaves 0.26 — which is exactly the churn rate of the dataset, its base rate.
A model that cannot separate the classes returns a group containing churners in the same
proportion as the population at large; picking the same number of users at random would
score identically. A working model is one whose selection is enriched, with precision
clearly above the base rate.

The first row deserves a second look: precision of 0.50 sounds respectable until the recall
is placed beside it — the model flagged seven users out of 2,400 and happened to be right
about three. No single metric is trustworthy on its own.

### 4. Feature engineering and the noise floor

**What is tested.** Feature engineering means building new columns out of existing ones —
`songs_per_minute = songs_played_per_day / listening_time` and similar. The question is
whether such a column can hold information about churn that the originals did not.

Theory answers: it cannot. A derived feature is computed from the originals, so anyone
holding the originals can produce it, and it therefore tells them nothing new. Formally this
is the data processing inequality, `I(f(X); Y) ≤ I(X; Y)`. Feature engineering does help in
practice, but by a different route: it presents information a model already had in a form
that model can actually use. It cannot create information. The experiment checks whether the
dataset is consistent with that expectation.

**How.** The measure is mutual information (MI) — how much knowing a feature sharpens
knowledge of the target. Zero means full independence: the feature value does not shift the
estimate of churn probability at all; larger values mean a stronger relationship.

The derived features are built in five ways typical for churn problems: an intensity ratio
`songs_per_minute`, an interaction `skip_x_songs`, an engagement measure `engagement`, and
the nonlinear transforms `time_squared` and `log_songs`. Their formulas involve exactly
three columns — `listening_time`, `songs_played_per_day`, `skip_rate`. The inequality
compares `f(X)` against the very `X` from which `f` was computed, so those three columns are
what serves as the original side — not all twenty: adding `age` or `country`, which appear
in none of the formulas, would make the comparison incomparable. It is closed — information
in the source columns against information in what was made from them.

**Result.**

```
Total MI, 3 original features:  0.00000
Total MI, 5 derived features:   0.01544
```

The derived set scores higher than the originals — precisely what the theorem forbids.

Before reading anything into that gap, it has to be established whether the method resolves
differences that small at all. That calls for data whose correct answer is known in advance,
built with the same device as experiment 2: shuffling the target. After shuffling, the true
MI is exactly zero, and anything the estimator reports above zero is pure error. The
measurement is taken on **the same five derived features**: each feature contributes its own
estimation error, a sum over five is noisier than a sum over three, so the configuration
that produced the suspicious number is the one that needs calibrating. Eight runs are used —
a single value shows no spread, and the number itself carries no significance; five or
fifteen would give the same picture.

```
8 runs: [0.00733 0.00054 0.00136 0.00921 0.02183 0.00890 0.00660 0.01458]
mean 0.00879, max 0.02183
```

**Conclusion.** On data guaranteed to hold no information the estimator still returns values
as high as 0.022: it works from a finite sample and reads coincidence as relationship. The
observed 0.01544 sits inside that range, so no measurement can separate it from zero. The
derived features added nothing, and the gap between 0.00000 and 0.01544 is entirely the
method's own imprecision.

The general rule follows: a small positive number proves nothing until it is compared
against the noise of the measurement itself.

### 5. The model learns — it just learns noise

**What is tested.** Null metrics might suggest that "no learning happens". That is not so:
learning happens, and very successfully. The experiment shows what exactly the model learns
and why only a held-out set makes it visible.

**How.** Models of increasing capacity are fitted on 70% of the data, then each is scored
twice: on the same 70% it has seen, and on the held-out 30% it has not.

**Result.**

| Trees | Depth | AUC (train) | AUC (held-out) |
|---|---|---|---|
| 50 | 2 | 0.5956 | 0.4810 |
| 200 | 3 | 0.7793 | 0.5059 |
| 500 | 5 | 0.9995 | 0.5190 |

![Training vs held-out performance](reports/figures/overfitting.png)

**Conclusion.** At 500 trees the model separates the training set almost perfectly: it has
enough capacity to memorise individual rows. Rules of the form *"the user with age 34,
listening_time 187 and skip_rate 0.22 churned"* are exactly true for one row and useless for
every other one, because no pattern stands behind them. On the held-out set the score stays
at coin-flip level.

The weakest configuration is the informative one: 50 trees at depth 2 cannot memorise, so
their training score of 0.5956 honestly reflects the absence of structure. The more capacity
a model has, the more convincingly it misleads on the training set — and the more essential
the held-out set becomes. Scored on training data alone, this dataset would look like a
complete success.

## Conclusion

Five independent lines of evidence agree:

- ROC-AUC of 0.50 across two model families and every preprocessing strategy
- results on real labels matching results on deliberately destroyed labels
- mutual information indistinguishable from the estimator's own noise floor
- precision pinned to the base rate regardless of balancing
- a tuned model that loses to a constant predictor by 0.007 F1

The features do not carry information about the target. The uniform, structureless
distributions of the numeric columns suggest the dataset is synthetic, with labels assigned
independently of user behaviour.

The practical takeaway is a habit: run a threshold-independent metric on a simple model
before starting any tuning. ROC-AUC near 0.50 means the ceiling has already been reached,
and further hyperparameter search will fit validation noise rather than improve anything.

## Repository structure

```
├── data/
│   └── spotify_churn_dataset.csv
├── reports/
│   └── figures/                 generated by the script
├── src/
│   └── experiments.py           all five experiments
├── requirements.txt
└── README.md
```

## Running

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/experiments.py
```

Runtime is roughly two minutes. All tables below are printed to stdout and the figures are
written to `reports/figures/`.

## Notes and limitations

- Reproduced on Python 3.14 with pandas 3.0.5, scikit-learn 1.9.0 and matplotlib 3.11.1.
- Absence of signal is demonstrated empirically, not proven. A relationship of a very
  different shape could exist and escape both tree ensembles and linear models — though
  the mutual information results make that unlikely.
- Extending this work would mean applying the same diagnostics to a dataset that does contain
  signal, so the two outcomes can be compared side by side.

## License

MIT — see [LICENSE](LICENSE).
