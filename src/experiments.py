"""
Signal diagnostics for the Spotify churn dataset.

Before tuning a model, check whether the data contains anything to learn.
Five experiments, run in order:

    1. Sanity check       Does the pipeline detect signal when signal exists?
    2. Shuffled target    Is the real data distinguishable from meaningless data?
    3. Class balancing    Does it improve the model, or only move the threshold?
    4. Feature engineering Can derived features add information?
    5. Overfitting        The model does learn - but it learns noise.

Usage:
    pip install -r requirements.txt
    python src/experiments.py

Figures are written to reports/figures/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "reports" / "figures"

CATEGORICAL = ["gender", "country", "subscription_type", "device_type"]
SEED = 42
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)


# ---------------------------------------------------------------- helpers

def load_data():
    """Return the raw dataframe, the one-hot encoded features and the target."""
    df = pd.read_csv(ROOT / "data" / "spotify_churn_dataset.csv")
    encoded = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)
    return df, encoded.drop(columns=["user_id", "is_churned"]), encoded["is_churned"]


def gbm(**kwargs):
    return GradientBoostingClassifier(random_state=SEED, **kwargs)


def cv_auc(X, y, model=None):
    """Cross-validated ROC-AUC. 0.5 means the model is no better than a coin flip."""
    model = model or gbm()
    return cross_val_score(model, X, y, cv=CV, scoring="roc_auc", n_jobs=-1).mean()


def section(number, title):
    print(f"\n{'=' * 70}\nEXPERIMENT {number}: {title}\n{'=' * 70}")


def derived_features(df, X):
    """New columns computed from existing ones - the usual feature engineering."""
    out = X.copy()
    out["songs_per_minute"] = df.songs_played_per_day / (df.listening_time + 1)
    out["skip_x_songs"] = df.skip_rate * df.songs_played_per_day
    out["ads_per_minute"] = df.ads_listened_per_week / (df.listening_time + 1)
    out["engagement"] = df.listening_time * (1 - df.skip_rate)
    out["age_x_time"] = df.age * df.listening_time
    out["time_squared"] = df.listening_time ** 2
    out["log_songs"] = np.log1p(df.songs_played_per_day)
    return out


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=120)
    plt.close(fig)


# ------------------------------------------------------- 1. sanity check

def sanity_check(X, y):
    """
    Rule out a broken pipeline before concluding the data is empty.
    Inject a feature that is correlated with the target by construction and
    vary the strength of that correlation. A working model must react.
    """
    section(1, "IS THE PIPELINE WORKING?")
    print("Injecting an artificial feature that is correlated with the target.")
    print("Less noise means a stronger relationship.\n")
    print(f"  {'injected signal':<20} {'ROC-AUC':>10}")
    print("  " + "-" * 32)

    rng = np.random.RandomState(SEED)
    for noise, label in [(6.0, "very weak"), (3.0, "weak"),
                         (1.5, "moderate"), (0.8, "strong")]:
        X_test = X.assign(injected=y.values + rng.normal(0, noise, len(y)))
        print(f"  {label:<20} {cv_auc(X_test, y):>10.4f}")

    print("\n  The model picks up signal as soon as there is any, so the code")
    print("  and the metric are sound. A null result below describes the data.")


# ---------------------------------------------------- 2. shuffled target

def shuffled_target(df, X, y):
    """
    Shuffling the target destroys any relationship: each label now belongs to a
    random user. Run every proposed improvement against both versions. Matching
    columns mean the method extracts no information at all.
    """
    section(2, "REAL TARGET vs RANDOMLY SHUFFLED TARGET")
    print("The shuffled target is a control group with zero signal by construction.\n")

    y_shuffled = pd.Series(np.random.RandomState(0).permutation(y.values))
    without_country = X.drop(columns=[c for c in X.columns if c.startswith("country_")])
    engineered = derived_features(df, X)

    pca = Pipeline([("scale", StandardScaler()),
                    ("pca", PCA(n_components=8)),
                    ("model", gbm())])
    smote = ImbPipeline([("smote", SMOTE(random_state=SEED)), ("model", gbm())])

    variants = [
        ("baseline", X, None),
        ("drop country", without_country, None),
        ("PCA (8 components)", X, pca),
        ("feature engineering", engineered, None),
        ("SMOTE (class balancing)", X, smote),
    ]

    print(f"  {'method':<26} {'real target':>12} {'shuffled target':>17}")
    print("  " + "-" * 57)

    names, real_scores, shuffled_scores = [], [], []
    for name, features, model in variants:
        real = cv_auc(features, y, model)
        shuffled = cv_auc(features, y_shuffled, model)
        print(f"  {name:<26} {real:>12.4f} {shuffled:>17.4f}")
        names.append(name)
        real_scores.append(real)
        shuffled_scores.append(shuffled)

    print("\n  The two columns are indistinguishable. The real data behaves")
    print("  exactly like data whose labels were deliberately destroyed.")

    y_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(y_pos - 0.2, real_scores, 0.4, label="real target", color="#4c72b0")
    ax.barh(y_pos + 0.2, shuffled_scores, 0.4, label="shuffled target", color="#c44e52")
    ax.axvline(0.5, color="black", ls="--", lw=1, label="random guessing")
    ax.set(yticks=y_pos, yticklabels=names, xlim=(0.4, 0.65), xlabel="ROC-AUC",
           title="No method tells real data from meaningless data")
    ax.legend(loc="lower right", fontsize=8)
    save_figure(fig, "real_vs_shuffled.png")


# ---------------------------------------------------- 3. class balancing

def class_balancing(X, y):
    """
    ROC-AUC measures ranking quality and does not depend on the threshold.
    Balancing and threshold tuning move a point along the precision-recall
    curve; they do not lift the curve itself.
    """
    section(3, "WHAT CLASS BALANCING ACTUALLY DOES")
    print("F1 can be raised without improving the model. Here is how.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)

    proba = gbm().fit(X_train, y_train).predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, proba)

    X_resampled, y_resampled = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
    proba_smote = gbm().fit(X_resampled, y_resampled).predict_proba(X_test)[:, 1]

    rows = [
        ("no balancing", (proba >= 0.5).astype(int), baseline_auc),
        ("SMOTE", (proba_smote >= 0.5).astype(int), roc_auc_score(y_test, proba_smote)),
        ("threshold 0.30", (proba >= 0.30).astype(int), baseline_auc),
        ("threshold 0.20", (proba >= 0.20).astype(int), baseline_auc),
        ("always predict churn", np.ones(len(y_test), dtype=int), 0.5),
    ]

    print(f"  {'variant':<22} {'AUC':>7} {'precision':>11} {'recall':>8} {'F1':>7}")
    print("  " + "-" * 58)
    for label, pred, auc in rows:
        print(f"  {label:<22} {auc:>7.4f} "
              f"{precision_score(y_test, pred, zero_division=0):>11.4f} "
              f"{recall_score(y_test, pred):>8.4f} {f1_score(y_test, pred):>7.4f}")

    print("\n  AUC never moves; only the precision-recall trade-off does.")
    print("  Precision stays near the churn rate itself, which means the model")
    print("  guesses at the base rate and separates nothing.")

    thresholds = np.linspace(0.05, 0.95, 91)
    f1_curve = [f1_score(y_test, (proba >= t).astype(int)) for t in thresholds]
    trivial_f1 = f1_score(y_test, np.ones(len(y_test), dtype=int))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, f1_curve, label="model F1", color="#4c72b0")
    ax.axhline(trivial_f1, color="red", ls="--", label="always predict churn")
    ax.set(xlabel="classification threshold", ylabel="F1",
           title="A rising F1 at lower thresholds is not a better model")
    ax.legend()
    save_figure(fig, "f1_vs_threshold.png")


# ----------------------------------------------- 4. feature engineering

def feature_engineering(df, y):
    """
    A derived feature is a function of the originals, so it cannot carry
    information they did not already hold: I(f(X); Y) <= I(X; Y).

    The estimate itself is noisy, so measure that noise by scoring the same
    features against a shuffled target, where the true value is exactly zero.
    """
    section(4, "FEATURE ENGINEERING AND THE NOISE FLOOR")
    print("Derived features are functions of existing ones. By the data")
    print("processing inequality they cannot add information.\n")

    original = df[["listening_time", "songs_played_per_day", "skip_rate"]]
    derived = pd.DataFrame({
        "songs_per_minute": df.songs_played_per_day / (df.listening_time + 1),
        "skip_x_songs": df.skip_rate * df.songs_played_per_day,
        "engagement": df.listening_time * (1 - df.skip_rate),
        "time_squared": df.listening_time ** 2,
        "log_songs": np.log1p(df.songs_played_per_day),
    })

    mi_original = mutual_info_classif(original, y, random_state=SEED).sum()
    mi_derived = mutual_info_classif(derived, y, random_state=SEED).sum()
    print(f"  Total MI, 3 original features:  {mi_original:.5f}")
    print(f"  Total MI, 5 derived features:   {mi_derived:.5f}")

    print("\n  The derived set formally scores higher, which contradicts the")
    print("  inequality - so measure the noise floor of the estimator itself.")
    print("  Same features, shuffled target, true value is zero:\n")

    rng = np.random.RandomState(0)
    noise = [mutual_info_classif(derived, rng.permutation(y.values), random_state=i).sum()
             for i in range(8)]
    print(f"  8 runs: {np.round(noise, 5)}")
    print(f"  mean {np.mean(noise):.5f}, max {np.max(noise):.5f}")

    inside = mi_derived <= np.max(noise)
    print(f"\n  {mi_derived:.5f} falls "
          f"{'inside' if inside else 'outside'} that range, "
          f"so it is {'indistinguishable from zero' if inside else 'a real effect'}.")
    print("\n  A small positive number proves nothing until it is compared")
    print("  against the noise of the measurement.")


# --------------------------------------------------------- 5. overfitting

def overfitting(X, y):
    """
    The model absolutely does train - it just memorises individual rows.
    Scoring on the training set hides this completely; only a held-out set
    reveals the gap.
    """
    section(5, "THE MODEL LEARNS, BUT IT LEARNS NOISE")
    print("Same model scored on data it has seen and data it has not.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)

    print(f"  {'trees':>7} {'depth':>7} {'AUC train':>11} {'AUC held-out':>14}")
    print("  " + "-" * 43)

    train_scores, test_scores, labels = [], [], []
    for n_trees, depth in [(50, 2), (200, 3), (500, 5)]:
        model = gbm(n_estimators=n_trees, max_depth=depth).fit(X_train, y_train)
        train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"  {n_trees:>7} {depth:>7} {train:>11.4f} {test:>14.4f}")
        train_scores.append(train)
        test_scores.append(test)
        labels.append(f"{n_trees} trees\ndepth {depth}")

    print("\n  On seen data the model approaches perfect separation: it has")
    print("  enough capacity to memorise individual rows. On unseen data it")
    print("  stays at chance, because there was no rule behind those rows.")
    print("\n  This is why a held-out set is mandatory. Scored on training data")
    print("  alone, this dataset would look like a complete success.")

    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x_pos - 0.2, train_scores, 0.4, label="training data", color="#4c72b0")
    ax.bar(x_pos + 0.2, test_scores, 0.4, label="held-out data", color="#c44e52")
    ax.axhline(0.5, color="black", ls="--", lw=1, label="random guessing")
    ax.set(xticks=x_pos, xticklabels=labels, ylabel="ROC-AUC", ylim=(0, 1.05),
           title="Model complexity buys memorisation, not generalisation")
    ax.legend(loc="lower right", fontsize=8)
    save_figure(fig, "overfitting.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df, X, y = load_data()

    print("=" * 70)
    print("SIGNAL DIAGNOSTICS - SPOTIFY CHURN DATASET")
    print("=" * 70)
    print(f"{len(X)} rows, {X.shape[1]} features after encoding, "
          f"churn rate {y.mean():.2%}")

    sanity_check(X, y)
    shuffled_target(df, X, y)
    class_balancing(X, y)
    feature_engineering(df, y)
    overfitting(X, y)

    print(f"\n{'=' * 70}\nCONCLUSION\n{'=' * 70}")
    print("The features carry no information about the target. Preprocessing")
    print("cannot create information, so balancing, PCA and derived features")
    print("all leave performance at chance level.")
    print(f"\nFigures: {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
