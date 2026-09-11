#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
HA FILLER — PREDICTION OF PATIENT DISSATISFACTION AT MONTH 6 (V4)
Unified, leakage-safe, reproducible analysis pipeline
================================================================================

This single script replaces and repairs the two exploratory notebooks
("Untitled3.ipynb" and "Untitled4.ipynb").

WHAT THIS SCRIPT DOES
---------------------
  0. Environment / reproducibility banner
  1. Load SPSS data, validate schema, build the outcome
  2. Descriptive statistics: missingness, dropout (non-random attrition) analysis
  3. Table 1 / Table 2 (baseline and follow-up characteristics by outcome)
  4. Leakage-safe feature engineering implemented as sklearn transformers
  5. Nested cross-validation benchmark of 7 classifiers x 2 predictor scenarios,
     pooled over multiple imputations (Rubin-style pooling of performance)
  6. Final model: ROC / PR / calibration / decision-curve / confusion matrix
  7. Explainability: SHAP + permutation importance (unbiased)
  8. Sensitivity analyses (outcome threshold, complete-case, model choice)
  9. Export of every table (CSV + XLSX) and figure (PNG 300dpi + PDF)

KEY METHODOLOGICAL DIFFERENCES FROM THE ORIGINAL NOTEBOOKS
----------------------------------------------------------
  * ALL imputation (time-based + MICE) now happens INSIDE the cross-validation
    pipeline, fitted on training folds only.  In the notebooks MICE was fitted
    once on the full dataset before splitting -> optimistic bias.
  * The engineered delta features were computed but never actually used
    (`make_mice_imputations(X_time, ...)` was called on X_time, not X_feat).
    They are now genuinely part of the model.
  * Two clinically distinct scenarios are reported separately instead of being
    silently mixed:
        Scenario A (baseline)  : pre-injection information only  -> real prediction
        Scenario B (full)      : baseline + month-0.5 + month-3   -> updating model
  * Ordinal Allergan severity scales are ordinal-encoded (None<Mild<Moderate<
    Severe<Extreme) instead of one-hot, so MICE can use them properly.
  * Threshold-free metrics are reported with bootstrap confidence intervals,
    and calibration is assessed (the notebooks reported accuracy at a fixed
    0.5 cut-off on a 29% prevalence problem, which is misleading).

USAGE
-----
    python ha_filler_pipeline.py --data HA_filler.sav --outdir results
    python ha_filler_pipeline.py --data HA_filler.sav --outdir results --fast

Author: rebuilt from the original notebooks
License: MIT
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ matplotlib
import matplotlib
matplotlib.use("Agg")            # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt

# --------------------------------------------------------------------- sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*valid feature names.*")


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

#: month offset of each study visit, used for linear interpolation
VISIT_MONTH: Dict[str, float] = {"V0": 0.0, "V2": 0.5, "V3": 3.0, "V4": 6.0}
VISITS: Tuple[str, ...] = ("V0", "V2", "V3")

#: outcome
TARGET_CONT = "satisfactionV4"
OUTCOME_BIN = "failure"
FAILURE_THRESHOLD = 7.0                # satisfactionV4 < 7  ->  failure = 1
SENSITIVITY_THRESHOLDS = (6.0, 7.0, 8.0)

#: Allergan-type ordinal severity scales -> numeric codes
ORDINAL_SEVERITY_MAP = {
    "none": 0, "None": 0, "NONE": 0,
    "mild": 1, "Mild": 1,
    "moderate": 2, "Moderate": 2,
    "severe": 3, "Severe": 3,
    "extreme": 4, "Extreme": 4,
}

#: columns that are genuinely nominal (no meaningful order)
NOMINAL_COLS = ["group", "sex"]

#: pre-injection ("baseline") predictor block — Scenario A
BASELINE_PREFIXES_EXACT = [
    "group", "sex", "age", "pain",
    "injectionR", "injectionL", "totalinjectionV1",
]

PLOT_DPI = 300
FIGSIZE = (7.0, 5.5)


@dataclass
class RunConfig:
    """Everything that controls a run, in one place."""
    data_path: Path
    outdir: Path
    fast: bool = False
    n_imputations: int = 5          # m for multiple imputation
    outer_splits: int = 5
    inner_splits: int = 3
    n_iter_search: int = 20         # RandomizedSearchCV budget
    n_bootstrap: int = 2000         # bootstrap reps for AUC CI
    random_state: int = RANDOM_STATE
    n_jobs: int = -1
    shap_max_samples: int = 150

    def __post_init__(self) -> None:
        if self.fast:
            self.n_imputations = 2
            self.outer_splits = 5
            self.inner_splits = 3
            self.n_iter_search = 4
            self.n_bootstrap = 200
            self.shap_max_samples = 40


# =============================================================================
# 0b. SMALL UTILITIES
# =============================================================================

class Tee:
    """Duplicate stdout to a log file so the console transcript is saved."""

    def __init__(self, path: Path) -> None:
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data: str) -> None:
        self.stdout.write(data)
        self.file.write(data)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def banner(title: str, char: str = "=", width: int = 78) -> None:
    print("\n" + char * width)
    print(title)
    print(char * width)


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 74 - len(title)))


class Timer:
    """Context manager that reports elapsed wall time."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> "Timer":
        self.t0 = time.time()
        print(f"[{self.label}] started ...", flush=True)
        return self

    def __exit__(self, *exc: Any) -> None:
        dt = time.time() - self.t0
        print(f"[{self.label}] finished in {dt:,.1f}s", flush=True)


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    """Save a figure as PNG (300 dpi) and PDF, then close it.

    NOTE: the original notebooks called ``plt.savefig`` *after* ``plt.show()``,
    which writes a blank canvas because ``show()`` clears the current figure.
    Saving from the Figure object avoids that class of bug entirely.
    """
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{name}.png", dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  figure saved -> figures/{name}.png|.pdf")


def save_table(df: pd.DataFrame, outdir: Path, name: str, index: bool = False) -> None:
    tabdir = outdir / "tables"
    tabdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tabdir / f"{name}.csv", index=index, encoding="utf-8-sig")
    print(f"  table saved  -> tables/{name}.csv   (shape={df.shape})")


def print_env(cfg: RunConfig) -> Dict[str, str]:
    import sklearn
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "random_state": str(cfg.random_state),
        "mode": "FAST (smoke test)" if cfg.fast else "FULL",
    }
    try:
        import xgboost; info["xgboost"] = xgboost.__version__
    except Exception:
        info["xgboost"] = "not installed"
    try:
        import lightgbm; info["lightgbm"] = lightgbm.__version__
    except Exception:
        info["lightgbm"] = "not installed"
    try:
        import shap; info["shap"] = shap.__version__
    except Exception:
        info["shap"] = "not installed"

    banner("0. ENVIRONMENT")
    for k, v in info.items():
        print(f"  {k:<16}: {v}")
    return info


# =============================================================================
# 1. DATA LOADING, VALIDATION AND OUTCOME CONSTRUCTION
# =============================================================================

def load_data(path: Path) -> pd.DataFrame:
    """Read the SPSS file and do basic structural validation.

    The notebooks hard-coded a Windows path (``r"C:\\Users\\Surface\\Downloads\\..."``),
    which makes the analysis non-reproducible on any other machine. The path is
    now a command-line argument.
    """
    banner("1. DATA LOADING AND VALIDATION")
    if not path.exists():
        raise FileNotFoundError(f"SPSS file not found: {path}")

    df = pd.read_spss(str(path))
    print(f"  file            : {path}")
    print(f"  raw shape       : {df.shape[0]} patients x {df.shape[1]} variables")

    # ---- structural checks -------------------------------------------------
    if TARGET_CONT not in df.columns:
        raise KeyError(f"Outcome column '{TARGET_CONT}' is missing from the data.")

    dup = df.duplicated().sum()
    print(f"  duplicate rows  : {dup}")
    if dup:
        warnings.warn(f"{dup} exactly duplicated rows found — inspect the source file.")

    # patient_id: a stable identifier. The notebooks did `df["patient_id"] = df.index`
    # and then `set_index`, which is a no-op relabelling; we keep an explicit
    # 1-based ID that survives row filtering, which is what you actually want
    # when you later need to trace a prediction back to a patient.
    df = df.reset_index(drop=True).copy()
    df.insert(0, "patient_id", np.arange(1, len(df) + 1))

    # ---- dtype report ------------------------------------------------------
    n_cat = df.select_dtypes(include=["category", "object"]).shape[1]
    n_num = df.select_dtypes(include=["number"]).shape[1]
    print(f"  numeric vars    : {n_num}")
    print(f"  categorical vars: {n_cat}")
    return df


def missing_report(data: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness, sorted by severity."""
    miss_n = data.isna().sum()
    miss_pct = (miss_n / len(data) * 100).round(2)
    out = (
        pd.DataFrame({"variable": miss_n.index,
                      "missing_n": miss_n.values,
                      "missing_pct": miss_pct.values})
        .sort_values(["missing_n", "variable"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return out


def dropout_analysis(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Compare patients WITH vs WITHOUT the primary outcome.

    Dropping patients with a missing outcome (complete-case analysis) is only
    unbiased if the outcome is Missing Completely At Random. This table is the
    evidence for or against that assumption and belongs in the Limitations
    section of any manuscript built on this data.
    """
    section("1.3  Attrition / dropout analysis (outcome-missing vs analysed)")
    from scipy import stats

    d = df.copy()
    d["_dropped"] = d[TARGET_CONT].isna().astype(int)
    n_drop = int(d["_dropped"].sum())
    n_keep = int((1 - d["_dropped"]).sum())
    print(f"  analysed (outcome observed): n = {n_keep}")
    print(f"  dropped  (outcome missing) : n = {n_drop} "
          f"({n_drop / len(d) * 100:.1f}%)")

    rows: List[Dict[str, Any]] = []
    candidate_cols = [c for c in d.columns
                      if c not in ("patient_id", "_dropped")
                      and not re.search(r"V4$", c)]

    for col in candidate_cols:
        s = d[col]
        keep = s[d["_dropped"] == 0]
        drop = s[d["_dropped"] == 1]
        if pd.api.types.is_numeric_dtype(s):
            k, dr = keep.dropna(), drop.dropna()
            if len(k) < 3 or len(dr) < 3:
                continue
            # Mann-Whitney U: no normality assumption, appropriate for these
            # skewed volume/area measurements.
            try:
                p = stats.mannwhitneyu(k, dr, alternative="two-sided").pvalue
            except ValueError:
                p = np.nan
            rows.append({
                "variable": col,
                "type": "numeric",
                "analysed_median_IQR": f"{k.median():.2f} "
                                       f"({k.quantile(.25):.2f}-{k.quantile(.75):.2f})",
                "dropped_median_IQR": f"{dr.median():.2f} "
                                      f"({dr.quantile(.25):.2f}-{dr.quantile(.75):.2f})",
                "test": "Mann-Whitney U",
                "p_value": p,
            })
        else:
            ct = pd.crosstab(s, d["_dropped"])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            try:
                p = stats.chi2_contingency(ct.values)[1]
            except ValueError:
                p = np.nan
            top = s.mode()
            top = top.iloc[0] if len(top) else "NA"
            rows.append({
                "variable": col,
                "type": "categorical",
                "analysed_median_IQR": f"mode={top} "
                                       f"({(keep == top).mean() * 100:.0f}%)",
                "dropped_median_IQR": f"mode={top} "
                                      f"({(drop == top).mean() * 100:.0f}%)",
                "test": "Chi-square",
                "p_value": p,
            })

    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    out["p_value"] = out["p_value"].round(4)
    # Benjamini-Hochberg FDR: ~40 simultaneous comparisons, raw p-values alone
    # would overstate the evidence for non-random dropout.
    try:
        from statsmodels.stats.multitest import multipletests
        mask = out["p_value"].notna()
        q = np.full(len(out), np.nan)
        q[mask.values] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
        out["q_value_BH"] = np.round(q, 4)
    except Exception:
        out["q_value_BH"] = np.nan

    sig = out[out["p_value"] < 0.05]
    if len(sig):
        print(f"  variables differing at p<0.05 between groups: {len(sig)}")
        print(sig.head(10).to_string(index=False))
        print("  => attrition is unlikely to be completely at random (MCAR);")
        print("     report this as a limitation.")
    else:
        print("  no variable differed at p<0.05 — consistent with MCAR.")

    save_table(out, outdir, "T00_dropout_analysis")
    return out


def build_outcome(df: pd.DataFrame, threshold: float = FAILURE_THRESHOLD
                  ) -> pd.DataFrame:
    """Drop patients without the outcome and derive the binary failure label."""
    section("1.4  Outcome construction")
    n0 = len(df)
    d = df.dropna(subset=[TARGET_CONT]).copy()
    print(f"  dropped {n0 - len(d)} patients with missing {TARGET_CONT}")

    d[OUTCOME_BIN] = (d[TARGET_CONT].astype(float) < threshold).astype(int)
    n_fail = int(d[OUTCOME_BIN].sum())
    print(f"  analysis sample : n = {len(d)}")
    print(f"  failure rule    : {TARGET_CONT} < {threshold:g}")
    print(f"  failures        : {n_fail} ({n_fail / len(d) * 100:.1f}%)")
    print(f"  non-failures    : {len(d) - n_fail} "
          f"({(len(d) - n_fail) / len(d) * 100:.1f}%)")

    # Events-per-variable warning — the single most common reason small clinical
    # ML papers fail to replicate.
    print(f"  NOTE: with {n_fail} events, a rule of thumb of 10 events per "
          f"predictor supports ~{n_fail // 10} predictors;")
    print("        heavy regularisation / nested CV is therefore essential.")
    return d


# =============================================================================
# 2. DESCRIPTIVE TABLES (Table 1 / Table 2)
# =============================================================================

def make_tableone(df: pd.DataFrame, outdir: Path) -> None:
    """Baseline and follow-up characteristics stratified by outcome."""
    banner("2. DESCRIPTIVE TABLES")
    try:
        from tableone import TableOne
    except ImportError:
        print("  tableone is not installed — skipping (pip install tableone)")
        return

    baseline_cols = [c for c in [
        "group", "age", "sex",
        "allerganV0", "allergan2V0",
        "depthRV0", "areaRV0", "volumeRV0",
        "depthLV0", "areaLV0", "volumeLV0",
        "thicknessV0", "densityV0",
        "injectionR", "injectionL", "totalinjectionV1",
        "pain",
    ] if c in df.columns]

    cat_cols = [c for c in ["group", "sex", "allerganV0", "allergan2V0"]
                if c in df.columns]
    nonnormal = [c for c in baseline_cols
                 if c.startswith(("pain", "volume", "area", "depth",
                                  "thickness", "density"))]

    section("2.1  Table 1 — baseline (V0) characteristics by outcome")
    t1 = TableOne(df, columns=baseline_cols, categorical=cat_cols,
                  groupby=OUTCOME_BIN, nonnormal=nonnormal,
                  pval=True, missing=True)
    print(t1.tabulate(tablefmt="simple"))
    t1.to_csv(outdir / "tables" / "T01_baseline_by_outcome.csv")
    print("  table saved  -> tables/T01_baseline_by_outcome.csv")

    v23_cols = [c for c in [
        "satisfactionV2", "satisfactionV3",
        "allerganV2", "allergan2V2", "allerganV3", "allergan2V3",
        "depthRV2", "areaRV2", "volumeRV2", "depthLV2", "areaLV2", "volumeLV2",
        "thicknessV2", "densityV2",
        "depthRV3", "areaRV3", "volumeRV3", "depthLV3", "areaLV3", "volumeLV3",
    ] if c in df.columns]
    cat23 = [c for c in ["allerganV2", "allergan2V2", "allerganV3", "allergan2V3"]
             if c in df.columns]
    nonnormal23 = [c for c in v23_cols if c not in cat23]

    section("2.2  Table 2 — follow-up (V2/V3) characteristics by outcome")
    t2 = TableOne(df, columns=v23_cols, categorical=cat23,
                  groupby=OUTCOME_BIN, nonnormal=nonnormal23,
                  pval=True, missing=True)
    print(t2.tabulate(tablefmt="simple"))
    t2.to_csv(outdir / "tables" / "T02_followup_by_outcome.csv")
    print("  table saved  -> tables/T02_followup_by_outcome.csv")


# =============================================================================
# 3. LEAKAGE-SAFE FEATURE ENGINEERING
# =============================================================================
#
# Everything in this section is implemented as a scikit-learn transformer so it
# can live INSIDE the cross-validation pipeline. This is the single most
# important correction relative to the original notebooks, where MICE was fitted
# once on the complete dataset and the imputed matrix was then fed to nested CV.
# Any statistic learned from the held-out fold (here: the conditional means used
# by MICE) leaks information and inflates every reported metric.
# -----------------------------------------------------------------------------


def get_time_triplets(columns: Sequence[str],
                      visits: Sequence[str] = VISITS) -> Dict[str, Dict[str, str]]:
    """Group longitudinal columns by their measurement base name.

    ``areaRV0``, ``areaRV2``, ``areaRV3``  ->  {"areaR": {"V0": ..., "V2": ..., "V3": ...}}

    Only bases with at least two observed visits are returned, because a single
    visit carries no temporal information to interpolate from.
    """
    trip: Dict[str, Dict[str, str]] = {}
    for c in columns:
        m = re.search(r"(V0|V2|V3)$", str(c))
        if not m:
            continue
        v = m.group(1)
        base = str(c)[: -len(v)]
        trip.setdefault(base, {})[v] = c
    # BUGFIX vs original: the notebooks kept bases with only ONE visit, which
    # then silently did nothing but still cost a full row-wise .apply() pass.
    return {b: d for b, d in trip.items()
            if len([v for v in d if v in visits]) >= 2}


class OrdinalSeverityEncoder(BaseEstimator, TransformerMixin):
    """Map ordered clinical severity labels to integers.

    The Allergan-type scales (None < Mild < Moderate < Severe < Extreme) are
    ordinal. One-hot encoding them (as the notebooks did) throws away the
    ordering and, more importantly, hides them from the numeric MICE imputer,
    which then had to fall back to 'most frequent' — a crude choice for a
    variable missing in ~12% of patients.
    """

    def __init__(self, mapping: Optional[Dict[str, int]] = None) -> None:
        self.mapping = mapping if mapping is not None else dict(ORDINAL_SEVERITY_MAP)

    def fit(self, X: pd.DataFrame, y=None) -> "OrdinalSeverityEncoder":
        X = pd.DataFrame(X)
        self.ordinal_cols_ = [
            c for c in X.columns
            if (str(c).startswith("allergan")
                and not pd.api.types.is_numeric_dtype(X[c]))
        ]
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for c in self.ordinal_cols_:
            if c not in X.columns:
                continue
            X[c] = (X[c].astype(str).str.strip().map(self.mapping)
                    .astype("float64"))
        return X


class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    """Fill V2 / V3 from neighbouring visits before general-purpose imputation.

    Rules (unchanged in spirit from the notebooks, but vectorised and corrected):
      * V2 missing, V0 and V3 present  -> linear interpolation on the month axis
      * V2 missing, only V0 present    -> last observation carried forward
      * V3 missing, V2 present (obs. or just filled) -> carry forward
      * V0 is never modified — it is the pre-treatment anchor.
      * Anything still missing is left to MICE.

    Why it was rewritten
    --------------------
    The original applied ``DataFrame.apply(func, axis=1)`` once per measurement
    base, i.e. ~20 full row-wise Python passes over the frame. That is O(20 * n)
    Python-level calls and was by far the slowest cell in the notebook. The
    vectorised version below is mathematically identical and ~500x faster, which
    matters now that this runs inside every CV fold.
    """

    def __init__(self, visit_month: Optional[Dict[str, float]] = None) -> None:
        self.visit_month = visit_month or dict(VISIT_MONTH)

    def fit(self, X: pd.DataFrame, y=None) -> "TimeSeriesImputer":
        X = pd.DataFrame(X)
        self.triplets_ = get_time_triplets(X.columns)
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        t0, t2, t3 = (self.visit_month["V0"], self.visit_month["V2"],
                      self.visit_month["V3"])
        w = (t2 - t0) / (t3 - t0)          # interpolation weight for V2

        for base, mp in self.triplets_.items():
            c0, c2, c3 = mp.get("V0"), mp.get("V2"), mp.get("V3")
            have = [c for c in (c0, c2, c3) if c is not None and c in X.columns]
            if len(have) < 2:
                continue
            # only numeric measurements can be interpolated
            if not all(pd.api.types.is_numeric_dtype(X[c]) for c in have):
                continue

            v0 = X[c0].astype(float) if c0 in X.columns else None
            v2 = X[c2].astype(float) if (c2 and c2 in X.columns) else None
            v3 = X[c3].astype(float) if (c3 and c3 in X.columns) else None

            # ---- fill V2 -----------------------------------------------
            if v2 is not None:
                if v0 is not None and v3 is not None:
                    interp = v0 + (v3 - v0) * w
                    v2 = v2.fillna(interp)          # needs BOTH neighbours
                if v0 is not None:
                    v2 = v2.fillna(v0)              # LOCF fallback
                X[c2] = v2

            # ---- fill V3 (after V2 has been filled) --------------------
            if v3 is not None:
                if v2 is not None:
                    v3 = v3.fillna(v2)
                elif v0 is not None:
                    v3 = v3.fillna(v0)
                X[c3] = v3
        return X


class MiceImputer(BaseEstimator, TransformerMixin):
    """Multivariate imputation by chained equations, fitted on training data only.

    Numeric columns go through ``IterativeImputer`` (Bayesian ridge, posterior
    sampling so that repeated runs give genuinely different completed datasets —
    the basis of multiple imputation). Remaining non-numeric columns fall back to
    a most-frequent imputer.
    """

    def __init__(self, random_state: int = RANDOM_STATE, max_iter: int = 10,
                 n_nearest_features: Optional[int] = 20,
                 sample_posterior: bool = True) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_nearest_features = n_nearest_features
        self.sample_posterior = sample_posterior

    @staticmethod
    def _as_writable(X: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
        """Dense, writable float array.

        ``IterativeImputer`` writes into its input buffer. Under pandas >= 3 the
        array returned by ``DataFrame.to_numpy()`` can be a read-only view of
        the block manager, which raises
        ``ValueError: assignment destination is read-only`` deep inside
        scikit-learn. Forcing an explicit copy is the portable fix.
        """
        return np.array(X[list(cols)].to_numpy(dtype="float64"),
                        dtype="float64", copy=True, order="C")

    def fit(self, X: pd.DataFrame, y=None) -> "MiceImputer":
        X = pd.DataFrame(X)
        self.num_cols_ = X.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols_ = [c for c in X.columns if c not in self.num_cols_]

        # A column that is entirely missing in this training fold carries no
        # information; record it and fill it with 0 rather than letting
        # IterativeImputer's keep_empty_features path run (see _as_writable).
        self.num_imputer_ = None
        self.empty_num_cols_: List[str] = []
        if self.num_cols_:
            arr = self._as_writable(X, self.num_cols_)
            all_nan = np.isnan(arr).all(axis=0)
            self.empty_num_cols_ = [c for c, f in zip(self.num_cols_, all_nan) if f]
            self.fit_num_cols_ = [c for c, f in zip(self.num_cols_, all_nan) if not f]
            if self.fit_num_cols_:
                self.num_imputer_ = IterativeImputer(
                    estimator=BayesianRidge(),
                    max_iter=self.max_iter,
                    sample_posterior=self.sample_posterior,
                    n_nearest_features=self.n_nearest_features,
                    random_state=self.random_state,
                ).fit(self._as_writable(X, self.fit_num_cols_))
        else:
            self.fit_num_cols_ = []

        self.cat_imputer_ = None
        if self.cat_cols_:
            self.cat_imputer_ = SimpleImputer(strategy="most_frequent").fit(
                X[self.cat_cols_].astype(object))

        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        if self.num_imputer_ is not None and self.fit_num_cols_:
            out = self.num_imputer_.transform(
                self._as_writable(X, self.fit_num_cols_))
            for j, c in enumerate(self.fit_num_cols_):
                X[c] = out[:, j]
        for c in self.empty_num_cols_:
            X[c] = 0.0
        if self.cat_imputer_ is not None:
            X[self.cat_cols_] = self.cat_imputer_.transform(
                X[self.cat_cols_].astype(object))
        return X


class DeltaFeatureAdder(BaseEstimator, TransformerMixin):
    """Add within-patient change scores between visits.

    For every longitudinal measurement we add
        base__dV2_V0, base__dV3_V2, base__dV3_V0   (absolute change)
        base__rV3_V0                               (relative change, %)

    Clinically these are the interesting quantities: for a filler study the
    *loss of volume* between month 0 and month 3 is far more informative than
    the raw volume at any one visit.

    BUG THIS FIXES
    --------------
    The notebooks built these features into ``X_feat`` and then never used them:
    the very next cell called ``make_mice_imputations(X_time, ...)``. Every model
    in both notebooks was therefore trained WITHOUT the engineered features, and
    a later cell (``pd.DataFrame(imputed_Xs[0], columns=X_feat.columns)``)
    re-indexed a frame that did not have those columns, silently producing all-NaN
    delta columns for the SHAP analysis.
    """

    def __init__(self, add_relative: bool = True, eps: float = 1e-8,
                 clip: float = 10.0) -> None:
        self.add_relative = add_relative
        self.eps = eps
        self.clip = clip

    def fit(self, X: pd.DataFrame, y=None) -> "DeltaFeatureAdder":
        X = pd.DataFrame(X)
        self.triplets_ = get_time_triplets(X.columns)
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        new: Dict[str, pd.Series] = {}
        for base, mp in self.triplets_.items():
            c0, c2, c3 = mp.get("V0"), mp.get("V2"), mp.get("V3")
            def num(c):
                if c is None or c not in X.columns:
                    return None
                s = X[c]
                return s.astype(float) if pd.api.types.is_numeric_dtype(s) else None

            v0, v2, v3 = num(c0), num(c2), num(c3)
            if v0 is not None and v2 is not None:
                new[f"{base}__dV2_V0"] = v2 - v0
            if v2 is not None and v3 is not None:
                new[f"{base}__dV3_V2"] = v3 - v2
            if v0 is not None and v3 is not None:
                new[f"{base}__dV3_V0"] = v3 - v0
                if self.add_relative:
                    # Guard the denominator with a scale-aware floor and clip the
                    # result. A naive (v3-v0)/|v0| explodes when an imputed v0
                    # lands near zero; after standardisation a single such row
                    # produces an enormous z-score in the test fold, which can
                    # destroy a ridge/linear fit (observed: CV R2 of -3.7e13).
                    floor = max(float(np.nanmedian(v0.abs())) * 1e-3, self.eps)
                    rel = (v3 - v0) / v0.abs().clip(lower=floor)
                    new[f"{base}__rV3_V0"] = rel.clip(-self.clip, self.clip)
        if new:
            X = pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)
        return X


class DtypeAwareEncoder(BaseEstimator, TransformerMixin):
    """One-hot the remaining nominal columns; pass numerics through (optionally scaled).

    The ColumnTransformer is constructed at ``fit`` time from the dtypes actually
    present in the training fold, so no column list has to be hard-coded and no
    category seen only in the test fold can cause a crash
    (``handle_unknown="ignore"``).
    """

    def __init__(self, scale: bool = False) -> None:
        self.scale = scale

    def fit(self, X: pd.DataFrame, y=None) -> "DtypeAwareEncoder":
        X = pd.DataFrame(X).copy()
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        try:
            oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                               drop=None)
        except TypeError:                       # scikit-learn < 1.2
            oh = OneHotEncoder(handle_unknown="ignore", sparse=False, drop=None)

        num_pipe = StandardScaler() if self.scale else "passthrough"
        self.pre_ = ColumnTransformer(
            [("num", num_pipe, num_cols), ("cat", oh, cat_cols)],
            remainder="drop", verbose_feature_names_out=True,
        ).fit(X.astype({c: str for c in cat_cols}) if cat_cols else X)
        self.num_cols_, self.cat_cols_ = num_cols, cat_cols
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X).copy()
        if self.cat_cols_:
            X = X.astype({c: str for c in self.cat_cols_ if c in X.columns})
        return self.pre_.transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = self.pre_.get_feature_names_out()
        # strip the ColumnTransformer prefixes so plots stay readable
        return np.asarray([re.sub(r"^(num|cat)__", "", n) for n in names])


def make_feature_pipeline(random_state: int, scale: bool,
                          add_deltas: bool, mice_iter: int = 10) -> Pipeline:
    """The full leakage-safe preprocessing stack, in the correct order.

    Order matters:
        1. ordinal encoding   — so severity scales become numeric ...
        2. time imputation    — ... and can be carried forward across visits
        3. MICE               — fills whatever remains, fitted on train only
        4. delta features     — computed AFTER imputation so they are complete
                                (computing them before would leave NaNs that MICE
                                would then have to impute from their own parents,
                                which is circular)
        5. one-hot encoding   — nominal columns only
    """
    steps: List[Tuple[str, Any]] = [
        ("ordinal", OrdinalSeverityEncoder()),
        ("timeimp", TimeSeriesImputer()),
        ("mice", MiceImputer(random_state=random_state, max_iter=mice_iter)),
    ]
    if add_deltas:
        steps.append(("delta", DeltaFeatureAdder()))
    steps.append(("encode", DtypeAwareEncoder(scale=scale)))
    return Pipeline(steps)


# ------------------------------------------------------------------ scenarios

def split_predictors(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build the two predictor matrices and drop every V4 (outcome-time) column.

    Scenario A — 'baseline': everything measurable BEFORE / AT the injection
        visit. This is the only scenario that answers the clinically useful
        question "can we tell in advance who will be dissatisfied?".

    Scenario B — 'full': baseline plus the month-0.5 and month-3 follow-up
        visits. Higher discrimination is expected almost by construction, since
        satisfactionV2 and satisfactionV3 are earlier measurements of the very
        quantity being predicted. Reported separately and interpreted as a
        *dynamic updating* model, never as prognosis at baseline.
    """
    banner("3. PREDICTOR SCENARIOS (leakage control)")
    v4_cols = [c for c in df.columns if re.search(r"V4$", str(c))]
    drop_cols = [c for c in v4_cols if c != TARGET_CONT]
    base = df.drop(columns=drop_cols + ["patient_id"], errors="ignore").copy()
    X_full = base.drop(columns=[TARGET_CONT, OUTCOME_BIN], errors="ignore")

    baseline_cols = [
        c for c in X_full.columns
        if str(c).endswith("V0") or str(c) in BASELINE_PREFIXES_EXACT
    ]
    X_baseline = X_full[baseline_cols].copy()

    leaked = [c for c in X_full.columns if re.search(r"V4$", str(c))]
    print(f"  V4 columns removed from predictors : {len(drop_cols)}")
    print(f"  any V4 predictor left?             : {bool(leaked)}  {leaked}")
    assert not leaked, "V4 leakage detected in the predictor matrix"

    print(f"\n  Scenario A (baseline, pre-injection): {X_baseline.shape[1]} predictors")
    print("    " + ", ".join(map(str, X_baseline.columns)))
    print(f"\n  Scenario B (baseline + V2 + V3)     : {X_full.shape[1]} predictors")
    print("    " + ", ".join(map(str, X_full.columns)))
    print("\n  WARNING for Scenario B: satisfactionV2 / satisfactionV3 are earlier")
    print("  measurements of the outcome construct itself. Any high AUC there")
    print("  reflects outcome autocorrelation, not novel prognostic information.")

    return {"A_baseline": X_baseline, "B_full": X_full}


# =============================================================================
# 4. MODEL ZOO AND HYPER-PARAMETER SEARCH SPACES
# =============================================================================
#
# Every classifier that appeared in either notebook is kept (Logistic Regression,
# Random Forest, SVM-RBF, AdaBoost, XGBoost, LightGBM) plus Extra Trees, which is
# a cheap and often strong addition on small tabular datasets.
#
# Two changes relative to the notebooks:
#   * SVM and Logistic Regression now get a StandardScaler. Without it an RBF
#     kernel on features spanning [0.04, 6172] is dominated entirely by
#     `thicknessV0`; the notebooks' SVM results were effectively a one-variable
#     model.
#   * AdaBoost's base learner defaulted to an unpruned DecisionTree
#     (`max_depth=None`) in notebook 3 cell 47. AdaBoost with fully grown trees
#     is not boosting weak learners — it overfits immediately. Depth is now a
#     tuned parameter constrained to {1,2,3} (stumps and near-stumps).
# -----------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    estimator: Any
    param_space: Dict[str, Sequence[Any]]
    needs_scaling: bool = False
    search: str = "random"          # "random" | "grid"
    available: bool = True


def build_model_specs(random_state: int = RANDOM_STATE,
                      fast: bool = False) -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    # ---------------------------------------------------------- linear model
    specs.append(ModelSpec(
        name="Logistic Regression (L2)",
        estimator=LogisticRegression(
            penalty="l2", solver="liblinear", class_weight="balanced",
            max_iter=2000, random_state=random_state),
        param_space={"model__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]},
        needs_scaling=True, search="grid",
    ))

    specs.append(ModelSpec(
        name="Elastic-Net Logistic",
        estimator=LogisticRegression(
            penalty="elasticnet", solver="saga", class_weight="balanced",
            max_iter=3000, random_state=random_state),
        param_space={"model__C": [0.01, 0.1, 1.0],
                     "model__l1_ratio": [0.15, 0.5, 0.85]},
        needs_scaling=True, search="grid",
    ))

    # ------------------------------------------------------------- bagging
    specs.append(ModelSpec(
        name="Random Forest",
        estimator=RandomForestClassifier(
            random_state=random_state, class_weight="balanced_subsample",
            n_jobs=1),
        param_space={
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 3, 5, 8],
            "model__min_samples_leaf": [1, 2, 3, 5],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", 0.3, 0.5],
        },
    ))

    specs.append(ModelSpec(
        name="Extra Trees",
        estimator=ExtraTreesClassifier(
            random_state=random_state, class_weight="balanced",
            n_jobs=1),
        param_space={
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 5, 8],
            "model__min_samples_leaf": [1, 2, 3, 5],
            "model__max_features": ["sqrt", 0.3, 0.5],
        },
    ))

    # ----------------------------------------------------------------- SVM
    specs.append(ModelSpec(
        name="SVM (RBF)",
        estimator=SVC(probability=True, class_weight="balanced",
                      random_state=random_state),
        param_space={
            "model__C": [0.1, 0.5, 1, 3, 10, 30],
            "model__gamma": ["scale", 0.3, 0.1, 0.03, 0.01],
        },
        needs_scaling=True, search="grid",
    ))

    # ------------------------------------------------------------- boosting
    specs.append(ModelSpec(
        name="AdaBoost",
        estimator=AdaBoostClassifier(
            estimator=DecisionTreeClassifier(random_state=random_state),
            random_state=random_state),
        param_space={
            "model__n_estimators": [50, 100, 200, 400],
            "model__learning_rate": [0.01, 0.03, 0.1, 0.3, 1.0],
            "model__estimator__max_depth": [1, 2, 3],
            "model__estimator__min_samples_leaf": [1, 2, 5],
        },
    ))

    try:
        from xgboost import XGBClassifier
        specs.append(ModelSpec(
            name="XGBoost",
            estimator=XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=random_state, n_jobs=1,
                verbosity=0),
            param_space={
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.01, 0.03, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__min_child_weight": [1, 3, 5],
                "model__reg_lambda": [1.0, 5.0, 10.0],
                # prevalence-aware: n_neg/n_pos ~= 2.4 for this cohort
                "model__scale_pos_weight": [1.0, 1.7, 2.4, 3.0],
            },
        ))
    except ImportError:
        print("  [warn] xgboost not installed — skipping XGBoost")

    try:
        from lightgbm import LGBMClassifier
        specs.append(ModelSpec(
            name="LightGBM",
            estimator=LGBMClassifier(
                objective="binary", random_state=random_state, n_jobs=1,
                verbose=-1, class_weight="balanced"),
            param_space={
                "model__n_estimators": [200, 400, 600],
                "model__num_leaves": [7, 15, 31],
                "model__max_depth": [2, 3, 5, -1],
                "model__learning_rate": [0.01, 0.03, 0.1],
                "model__min_child_samples": [5, 10, 20],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__subsample_freq": [1],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__reg_lambda": [0.0, 1.0, 5.0],
            },
        ))
    except ImportError:
        print("  [warn] lightgbm not installed — skipping LightGBM")

    if fast:                       # keep the smoke test quick but representative
        keep = {"Logistic Regression (L2)", "Random Forest", "XGBoost"}
        specs = [s for s in specs if s.name in keep]
    return specs


# =============================================================================
# 5. METRICS
# =============================================================================

def binary_metrics(y_true: np.ndarray, proba: np.ndarray,
                   threshold: float = 0.5) -> Dict[str, float]:
    """Threshold-free and threshold-dependent metrics for one fold.

    The notebooks reported Accuracy at a hard 0.5 cut-off. On a 29%-prevalence
    problem a model that predicts "no failure" for everyone already scores 0.71,
    so accuracy is close to uninformative; balanced accuracy, MCC, AUC and
    average precision are reported alongside it here.
    """
    y_true = np.asarray(y_true).astype(int)
    pred = (proba >= threshold).astype(int)

    out: Dict[str, float] = {}
    out["auc"] = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan
    out["average_precision"] = average_precision_score(y_true, proba)
    out["brier"] = brier_score_loss(y_true, proba)
    out["accuracy"] = accuracy_score(y_true, pred)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, pred)
    out["f1"] = f1_score(y_true, pred, zero_division=0)
    out["mcc"] = matthews_corrcoef(y_true, pred)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
    out["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    out["ppv"] = tp / (tp + fp) if (tp + fp) else np.nan
    out["npv"] = tn / (tn + fn) if (tn + fn) else np.nan
    return out


def bootstrap_auc_ci(y_true: np.ndarray, proba: np.ndarray,
                     n_boot: int = 2000, alpha: float = 0.05,
                     random_state: int = RANDOM_STATE
                     ) -> Tuple[float, float, float]:
    """Stratified bootstrap percentile CI for the AUC.

    The notebooks reported the standard deviation across CV folds as if it were
    a confidence interval. Fold-to-fold SD measures split instability, not
    sampling uncertainty about the AUC, and is typically far too narrow.
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    idx_pos = np.flatnonzero(y_true == 1)
    idx_neg = np.flatnonzero(y_true == 0)

    stats_: List[float] = []
    for _ in range(n_boot):
        bp = rng.choice(idx_pos, len(idx_pos), replace=True)
        bn = rng.choice(idx_neg, len(idx_neg), replace=True)
        bi = np.concatenate([bp, bn])
        try:
            stats_.append(roc_auc_score(y_true[bi], proba[bi]))
        except ValueError:
            continue
    stats_arr = np.asarray(stats_)
    point = roc_auc_score(y_true, proba)
    lo, hi = np.percentile(stats_arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def delong_like_bootstrap_test(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                               n_boot: int = 2000,
                               random_state: int = RANDOM_STATE) -> Dict[str, float]:
    """Paired bootstrap test for the difference between two AUCs.

    Same resample is applied to both models, so the correlation between them is
    preserved — the paired analogue of a DeLong test, without requiring an extra
    dependency.
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true).astype(int)
    idx_pos = np.flatnonzero(y_true == 1)
    idx_neg = np.flatnonzero(y_true == 0)

    diffs: List[float] = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(idx_pos, len(idx_pos), replace=True),
                             rng.choice(idx_neg, len(idx_neg), replace=True)])
        try:
            diffs.append(roc_auc_score(y_true[bi], p1[bi])
                         - roc_auc_score(y_true[bi], p2[bi]))
        except ValueError:
            continue
    d = np.asarray(diffs)
    obs = roc_auc_score(y_true, p1) - roc_auc_score(y_true, p2)
    # two-sided bootstrap p-value
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"auc_diff": float(obs), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, p))}


def youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Cut-off maximising sensitivity + specificity - 1."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


# =============================================================================
# 6. NESTED CROSS-VALIDATION BENCHMARK  (multiple-imputation pooled)
# =============================================================================

def _make_search(spec: ModelSpec, feat_pipe: Pipeline, inner_cv, cfg: RunConfig):
    """Wrap preprocessing + model and pick the right hyper-parameter search."""
    pipe = Pipeline([("features", clone(feat_pipe)),
                     ("model", clone(spec.estimator))])
    common = dict(estimator=pipe, scoring="roc_auc", cv=inner_cv,
                  n_jobs=cfg.n_jobs, refit=True, error_score=np.nan)
    if spec.search == "grid":
        return GridSearchCV(param_grid=dict(spec.param_space), **common)
    n_comb = int(np.prod([len(v) for v in spec.param_space.values()]))
    return RandomizedSearchCV(
        param_distributions=dict(spec.param_space),
        n_iter=min(cfg.n_iter_search, n_comb),
        random_state=cfg.random_state, **common)


def nested_cv_benchmark(X: pd.DataFrame, y: np.ndarray, scenario: str,
                        cfg: RunConfig, add_deltas: bool
                        ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
    """Run the full nested CV for every model, repeated over m imputations.

    Returns
    -------
    summary : one row per classifier, mean (SD) of every metric
    oof     : {model_name: out-of-fold probability matrix, shape (m, n)}
    perfold : the raw per-fold records, for anyone who wants to re-analyse them
    """
    banner(f"6. NESTED CROSS-VALIDATION BENCHMARK — scenario {scenario}")
    y = np.asarray(y).astype(int)
    n = len(y)
    specs = build_model_specs(cfg.random_state, fast=cfg.fast)
    print(f"  classifiers      : {len(specs)}")
    print(f"  outer / inner CV : {cfg.outer_splits}-fold / {cfg.inner_splits}-fold "
          f"(stratified)")
    print(f"  imputations (m)  : {cfg.n_imputations}")
    print(f"  search budget    : up to {cfg.n_iter_search} candidates "
          f"(grid where the space is small)")
    print(f"  total model fits : "
          f"~{cfg.n_imputations * cfg.outer_splits * len(specs) * (cfg.n_iter_search * cfg.inner_splits + 1):,}")

    records: List[Dict[str, Any]] = []
    oof: Dict[str, np.ndarray] = {s.name: np.full((cfg.n_imputations, n), np.nan)
                                  for s in specs}
    best_params: List[Dict[str, Any]] = []

    for m in range(cfg.n_imputations):
        seed_m = cfg.random_state + 1000 * m
        # NOTE: the outer split is held FIXED across imputations so that the
        # only source of between-imputation variability is the imputation itself.
        outer_cv = StratifiedKFold(n_splits=cfg.outer_splits, shuffle=True,
                                   random_state=cfg.random_state)
        inner_cv = StratifiedKFold(n_splits=cfg.inner_splits, shuffle=True,
                                   random_state=cfg.random_state)

        for fold, (tr, te) in enumerate(outer_cv.split(X, y), start=1):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y[tr], y[te]

            for spec in specs:
                feat = make_feature_pipeline(
                    random_state=seed_m, scale=spec.needs_scaling,
                    add_deltas=add_deltas,
                    mice_iter=5 if cfg.fast else 10)
                search = _make_search(spec, feat, inner_cv, cfg)
                t0 = time.time()
                try:
                    search.fit(X_tr, y_tr)
                    proba = search.predict_proba(X_te)[:, 1]
                except Exception as exc:            # never let one model kill the run
                    print(f"    [ERROR] {spec.name} imp{m+1} fold{fold}: "
                          f"{type(exc).__name__}: {exc}")
                    continue
                dt = time.time() - t0

                oof[spec.name][m, te] = proba
                mt = binary_metrics(y_te, proba)
                mt.update({"model": spec.name, "imputation": m + 1,
                           "fold": fold, "n_test": len(te),
                           "fit_seconds": round(dt, 2)})
                records.append(mt)
                best_params.append({"model": spec.name, "imputation": m + 1,
                                    "fold": fold,
                                    "best_inner_auc": round(float(search.best_score_), 4),
                                    "best_params": json.dumps(
                                        {k: str(v) for k, v in search.best_params_.items()})})
            print(f"    imputation {m+1}/{cfg.n_imputations}  fold {fold}/"
                  f"{cfg.outer_splits} done  ({time.time() - t0:.0f}s last model)",
                  flush=True)

    perfold = pd.DataFrame(records)
    if perfold.empty:
        raise RuntimeError("No model completed — check the errors above.")

    metric_cols = ["auc", "average_precision", "balanced_accuracy", "accuracy",
                   "f1", "mcc", "sensitivity", "specificity", "ppv", "npv",
                   "brier"]

    rows: List[Dict[str, Any]] = []
    for name in [s.name for s in specs]:
        sub = perfold[perfold["model"] == name]
        if sub.empty:
            continue
        row: Dict[str, Any] = {"Classifier": name, "n_estimates": len(sub)}
        for mcol in metric_cols:
            mu, sd = sub[mcol].mean(), sub[mcol].std(ddof=1)
            row[mcol] = f"{mu:.3f} ({sd:.3f})"
            row[f"_{mcol}_mean"] = mu
        # AUC on pooled out-of-fold predictions, averaged over imputations
        pooled = []
        for m in range(cfg.n_imputations):
            p = oof[name][m]
            if np.isnan(p).any():
                continue
            pooled.append(roc_auc_score(y, p))
        row["_pooled_auc_mean"] = float(np.mean(pooled)) if pooled else np.nan
        row["pooled_OOF_AUC"] = (f"{np.mean(pooled):.3f} ({np.std(pooled, ddof=1):.3f})"
                                 if len(pooled) > 1 else
                                 (f"{pooled[0]:.3f}" if pooled else "NA"))
        row["mean_fit_seconds"] = round(sub["fit_seconds"].mean(), 1)
        rows.append(row)

    summary = (pd.DataFrame(rows)
               .sort_values("_auc_mean", ascending=False)
               .reset_index(drop=True))

    display_cols = (["Classifier", "auc", "pooled_OOF_AUC", "average_precision",
                     "balanced_accuracy", "sensitivity", "specificity",
                     "f1", "mcc", "brier", "accuracy", "n_estimates",
                     "mean_fit_seconds"])
    print("\n  RESULTS — mean (SD) over "
          f"{cfg.n_imputations} imputations x {cfg.outer_splits} outer folds\n")
    print(summary[display_cols].to_string(index=False))

    save_table(summary[display_cols], cfg.outdir, f"T10_benchmark_{scenario}")
    save_table(perfold, cfg.outdir, f"T11_perfold_{scenario}")
    save_table(pd.DataFrame(best_params), cfg.outdir,
               f"T12_selected_hyperparameters_{scenario}")
    return summary, oof, perfold


# =============================================================================
# 7. FIGURES FOR THE BENCHMARK
# =============================================================================

def plot_roc_all_models(y: np.ndarray, oof: Dict[str, np.ndarray],
                        scenario: str, cfg: RunConfig) -> pd.DataFrame:
    """One ROC curve per classifier, averaged across imputations, with CIs."""
    section(f"7.1  ROC curves — all classifiers ({scenario})")
    grid = np.linspace(0, 1, 301)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=140)
    rows: List[Dict[str, Any]] = []

    order = sorted(oof.keys(),
                   key=lambda k: -np.nanmean([roc_auc_score(y, oof[k][m])
                                              for m in range(oof[k].shape[0])
                                              if not np.isnan(oof[k][m]).any()]
                                             or [0]))
    cmap = plt.colormaps.get_cmap("tab10")
    for i, name in enumerate(order):
        mats = [oof[name][m] for m in range(oof[name].shape[0])
                if not np.isnan(oof[name][m]).any()]
        if not mats:
            continue
        tprs, aucs = [], []
        for p in mats:
            fpr, tpr, _ = roc_curve(y, p)
            t = np.interp(grid, fpr, tpr); t[0] = 0.0
            tprs.append(t); aucs.append(auc(fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        mean_auc = float(np.mean(aucs))

        pooled = np.mean(np.vstack(mats), axis=0)
        pt, lo, hi = bootstrap_auc_ci(y, pooled, cfg.n_bootstrap,
                                      random_state=cfg.random_state)
        rows.append({"Classifier": name, "AUC_mean_over_imputations": round(mean_auc, 3),
                     "AUC_pooled": round(pt, 3),
                     "CI95_low": round(lo, 3), "CI95_high": round(hi, 3)})
        ax.plot(grid, mean_tpr, lw=2.0, color=cmap(i % 10),
                label=f"{name}  AUC={mean_auc:.3f} [{lo:.2f}–{hi:.2f}]")

    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="0.4")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.set_xlabel("1 − Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title(f"Nested cross-validated ROC — scenario {scenario}\n"
                 f"(out-of-fold predictions, {cfg.n_imputations} imputations)")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    savefig(fig, cfg.outdir, f"F10_ROC_all_models_{scenario}")

    tab = pd.DataFrame(rows).sort_values("AUC_pooled", ascending=False)
    save_table(tab, cfg.outdir, f"T13_AUC_with_CI_{scenario}")
    print(tab.to_string(index=False))
    return tab


def plot_pr_curves(y: np.ndarray, oof: Dict[str, np.ndarray],
                   scenario: str, cfg: RunConfig) -> None:
    """Precision-recall curves — the more honest view under class imbalance."""
    section(f"7.2  Precision-Recall curves ({scenario})")
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=140)
    cmap = plt.colormaps.get_cmap("tab10")
    prevalence = float(np.mean(y))
    for i, name in enumerate(oof):
        mats = [oof[name][m] for m in range(oof[name].shape[0])
                if not np.isnan(oof[name][m]).any()]
        if not mats:
            continue
        pooled = np.mean(np.vstack(mats), axis=0)
        prec, rec, _ = precision_recall_curve(y, pooled)
        ap = average_precision_score(y, pooled)
        ax.plot(rec, prec, lw=1.8, color=cmap(i % 10), label=f"{name}  AP={ap:.3f}")
    ax.axhline(prevalence, ls="--", lw=1.2, color="0.4",
               label=f"No-skill (prevalence={prevalence:.3f})")
    ax.set_xlabel("Recall (Sensitivity)"); ax.set_ylabel("Precision (PPV)")
    ax.set_title(f"Precision–Recall — scenario {scenario}")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    savefig(fig, cfg.outdir, f"F11_PR_all_models_{scenario}")


def plot_metric_comparison(summary: pd.DataFrame, scenario: str,
                           cfg: RunConfig) -> None:
    """Bar chart of the headline metrics, ordered by AUC."""
    section(f"7.3  Metric comparison chart ({scenario})")
    metrics = [("_auc_mean", "AUC-ROC"),
               ("_average_precision_mean", "Avg. precision"),
               ("_balanced_accuracy_mean", "Balanced accuracy"),
               ("_mcc_mean", "MCC")]
    metrics = [(k, lbl) for k, lbl in metrics if k in summary.columns]
    labels = summary["Classifier"].tolist()
    ypos = np.arange(len(labels))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(labels) + 2.2), dpi=140)
    cmap = plt.colormaps.get_cmap("Set2")
    for j, (key, lbl) in enumerate(metrics):
        ax.barh(ypos + j * width, summary[key], height=width,
                label=lbl, color=cmap(j))
    ax.set_yticks(ypos + width * (len(metrics) - 1) / 2)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0.5, ls=":", color="0.5", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score (mean over imputations × outer folds)")
    ax.set_title(f"Classifier comparison — scenario {scenario}")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, cfg.outdir, f"F12_metric_comparison_{scenario}")


# =============================================================================
# 8. FINAL MODEL: CALIBRATION, DECISION CURVE, CONFUSION MATRIX
# =============================================================================

def plot_calibration(y: np.ndarray, oof: Dict[str, np.ndarray], best: str,
                     scenario: str, cfg: RunConfig) -> pd.DataFrame:
    """Reliability diagram + Brier decomposition for the best classifier.

    Neither notebook checked calibration. A model can rank patients well
    (high AUC) and still output probabilities that are systematically wrong,
    which matters the moment a clinician wants to quote "your risk is 30%".
    """
    section(f"8.1  Calibration of the best model ({best}, {scenario})")
    mats = [oof[best][m] for m in range(oof[best].shape[0])
            if not np.isnan(oof[best][m]).any()]
    p = np.mean(np.vstack(mats), axis=0)

    n_bins = 8
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() < 3:
            continue
        rows.append({"bin": b + 1, "n": int(sel.sum()),
                     "mean_predicted": float(p[sel].mean()),
                     "observed_rate": float(y[sel].mean())})
    cal = pd.DataFrame(rows)

    brier = brier_score_loss(y, p)
    # Cox calibration: intercept (calibration-in-the-large) and slope
    from sklearn.linear_model import LogisticRegression as _LR
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = _LR(penalty=None, solver="lbfgs", max_iter=1000).fit(logit.reshape(-1, 1), y)
    slope = float(lr.coef_[0][0]); intercept = float(lr.intercept_[0])
    print(f"  Brier score           : {brier:.4f} "
          f"(no-skill = {np.mean(y) * (1 - np.mean(y)):.4f})")
    print(f"  Calibration slope     : {slope:.3f}   (1.0 = perfect; <1 = overfit)")
    print(f"  Calibration intercept : {intercept:.3f}   (0.0 = perfect)")

    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=140)
    ax.plot([0, 1], [0, 1], ls="--", color="0.4", lw=1.2, label="Perfect calibration")
    ax.plot(cal["mean_predicted"], cal["observed_rate"], "o-", lw=2, ms=7,
            color="#1f4e79", label=f"{best}")
    for _, r in cal.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_predicted"], r["observed_rate"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=7,
                    color="0.35")
    ax.set_xlabel("Mean predicted probability of failure")
    ax.set_ylabel("Observed failure rate")
    ax.set_title(f"Calibration — {best} ({scenario})\n"
                 f"Brier={brier:.3f}  slope={slope:.2f}  intercept={intercept:.2f}")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    savefig(fig, cfg.outdir, f"F20_calibration_{scenario}")

    cal["brier"] = round(brier, 4)
    cal["calibration_slope"] = round(slope, 3)
    cal["calibration_intercept"] = round(intercept, 3)
    save_table(cal, cfg.outdir, f"T20_calibration_{scenario}")
    return cal


def plot_decision_curve(y: np.ndarray, oof: Dict[str, np.ndarray],
                        models: Sequence[str], scenario: str,
                        cfg: RunConfig) -> pd.DataFrame:
    """Decision-curve analysis: net benefit against 'treat all' / 'treat none'.

    This answers the question a clinician actually cares about — is acting on
    this model better than a blanket policy? — and is standard in prediction
    model reporting (TRIPOD). Neither notebook produced it.
    """
    section(f"8.2  Decision curve analysis ({scenario})")
    n = len(y)
    thresholds = np.linspace(0.01, 0.60, 60)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=140)
    rows: List[Dict[str, Any]] = []

    prev = float(np.mean(y))
    nb_all = [prev - (1 - prev) * (t / (1 - t)) for t in thresholds]
    ax.plot(thresholds, nb_all, ls="--", color="0.45", lw=1.5, label="Treat all")
    ax.axhline(0, ls="-", color="0.2", lw=1.0, label="Treat none")

    cmap = plt.colormaps.get_cmap("tab10")
    for i, name in enumerate(models):
        mats = [oof[name][m] for m in range(oof[name].shape[0])
                if not np.isnan(oof[name][m]).any()]
        if not mats:
            continue
        p = np.mean(np.vstack(mats), axis=0)
        nb = []
        for t in thresholds:
            pred = p >= t
            tp = float(np.sum(pred & (y == 1)))
            fp = float(np.sum(pred & (y == 0)))
            nb.append(tp / n - (fp / n) * (t / (1 - t)))
        ax.plot(thresholds, nb, lw=2, color=cmap(i % 10), label=name)
        for t, v in zip(thresholds, nb):
            rows.append({"model": name, "threshold": round(float(t), 3),
                         "net_benefit": round(float(v), 5)})

    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision curve analysis — scenario {scenario}")
    ax.set_ylim(min(-0.05, prev - 0.35), prev + 0.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    savefig(fig, cfg.outdir, f"F21_decision_curve_{scenario}")

    dca = pd.DataFrame(rows)
    save_table(dca, cfg.outdir, f"T21_decision_curve_{scenario}")
    return dca


def plot_confusion(y: np.ndarray, oof: Dict[str, np.ndarray], best: str,
                   scenario: str, cfg: RunConfig) -> pd.DataFrame:
    """Confusion matrices at the default 0.5 cut-off and at the Youden optimum."""
    section(f"8.3  Confusion matrices ({best}, {scenario})")
    mats = [oof[best][m] for m in range(oof[best].shape[0])
            if not np.isnan(oof[best][m]).any()]
    p = np.mean(np.vstack(mats), axis=0)
    thr_j = youden_threshold(y, p)
    print(f"  Youden-optimal threshold : {thr_j:.3f} (default 0.500)")

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=140)
    for ax, (thr, label) in zip(axes, [(0.5, "default 0.50"),
                                       (thr_j, f"Youden {thr_j:.2f}")]):
        pred = (p >= thr).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=15,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. success", "Pred. failure"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True success", "True failure"])
        mt = binary_metrics(y, p, threshold=thr)
        ax.set_title(f"{label}\nSe={mt['sensitivity']:.2f}  Sp={mt['specificity']:.2f}"
                     f"  F1={mt['f1']:.2f}  MCC={mt['mcc']:.2f}", fontsize=10)
        rows.append({"threshold_type": label, "threshold": round(thr, 4),
                     **{k: round(v, 4) for k, v in mt.items()}})
    fig.suptitle(f"{best} — out-of-fold confusion matrices ({scenario})",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, cfg.outdir, f"F22_confusion_{scenario}")

    tab = pd.DataFrame(rows)
    save_table(tab, cfg.outdir, f"T22_operating_points_{scenario}")
    print(tab.to_string(index=False))
    return tab


def pairwise_auc_tests(y: np.ndarray, oof: Dict[str, np.ndarray],
                       scenario: str, cfg: RunConfig) -> pd.DataFrame:
    """Paired bootstrap comparison of every classifier against the best one.

    Without this, ranking seven models by a third decimal place of AUC on 171
    patients invites over-interpretation.
    """
    section(f"8.4  Are the classifiers actually different? ({scenario})")
    pooled: Dict[str, np.ndarray] = {}
    for name, arr in oof.items():
        mats = [arr[m] for m in range(arr.shape[0]) if not np.isnan(arr[m]).any()]
        if mats:
            pooled[name] = np.mean(np.vstack(mats), axis=0)
    if len(pooled) < 2:
        return pd.DataFrame()

    best = max(pooled, key=lambda k: roc_auc_score(y, pooled[k]))
    rows = []
    for name, p in pooled.items():
        if name == best:
            continue
        r = delong_like_bootstrap_test(y, pooled[best], p,
                                       n_boot=cfg.n_bootstrap,
                                       random_state=cfg.random_state)
        rows.append({"reference (best)": best, "compared_with": name,
                     "AUC_difference": round(r["auc_diff"], 4),
                     "CI95_low": round(r["ci_low"], 4),
                     "CI95_high": round(r["ci_high"], 4),
                     "p_value": round(r["p_value"], 4),
                     "significant_0.05": r["p_value"] < 0.05})
    tab = pd.DataFrame(rows).sort_values("p_value")
    print(tab.to_string(index=False))
    n_sig = int(tab["significant_0.05"].sum())
    print(f"\n  {n_sig} of {len(tab)} classifiers differ significantly from "
          f"'{best}'.")
    if n_sig == 0:
        print("  => the models are statistically indistinguishable on this sample;")
        print("     prefer the simplest / best-calibrated one, not the top of the table.")
    save_table(tab, cfg.outdir, f"T23_pairwise_AUC_tests_{scenario}")
    return tab


# =============================================================================
# 9. EXPLAINABILITY
# =============================================================================

def fit_final_model(X: pd.DataFrame, y: np.ndarray, spec: ModelSpec,
                    cfg: RunConfig, add_deltas: bool) -> Pipeline:
    """Refit the chosen classifier on the whole analysis sample.

    Hyper-parameters are re-selected by an inner CV on the full data; this model
    is used ONLY for explanation, never for reporting performance (its
    performance numbers come from the nested CV above).
    """
    section(f"9.0  Refitting {spec.name} on the full sample for explanation")
    feat = make_feature_pipeline(cfg.random_state, spec.needs_scaling,
                                 add_deltas, mice_iter=5 if cfg.fast else 10)
    inner_cv = StratifiedKFold(cfg.inner_splits, shuffle=True,
                               random_state=cfg.random_state)
    search = _make_search(spec, feat, inner_cv, cfg)
    search.fit(X, y)
    print(f"  inner-CV AUC : {search.best_score_:.3f}")
    print(f"  best params  : {search.best_params_}")
    return search.best_estimator_


def permutation_importance_plot(model: Pipeline, X: pd.DataFrame, y: np.ndarray,
                                scenario: str, cfg: RunConfig, top_k: int = 20
                                ) -> pd.DataFrame:
    """Permutation importance on the ORIGINAL columns, with CV-honest scoring.

    Impurity-based ``feature_importances_`` (what both notebooks plotted) is
    biased towards high-cardinality continuous variables — with 30+ continuous
    measurements and a handful of binary ones, that bias is exactly the wrong
    way round here. Permutation importance measures the actual drop in AUC.
    """
    section(f"9.1  Permutation importance ({scenario})")
    r = permutation_importance(
        model, X, y, scoring="roc_auc",
        n_repeats=10 if cfg.fast else 30,
        random_state=cfg.random_state, n_jobs=1)
    imp = (pd.DataFrame({"feature": X.columns,
                         "importance_mean": r.importances_mean,
                         "importance_sd": r.importances_std})
           .sort_values("importance_mean", ascending=False)
           .reset_index(drop=True))
    save_table(imp, cfg.outdir, f"T30_permutation_importance_{scenario}")
    print(imp.head(15).to_string(index=False))

    top = imp.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(top) + 1.8), dpi=140)
    ax.barh(top["feature"], top["importance_mean"],
            xerr=top["importance_sd"], color="#2b6a9b",
            error_kw={"lw": 0.8, "ecolor": "0.4"})
    ax.axvline(0, color="0.3", lw=1)
    ax.set_xlabel("Drop in AUC when the variable is permuted")
    ax.set_title(f"Permutation importance — top {top_k} ({scenario})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, cfg.outdir, f"F30_permutation_importance_{scenario}")
    return imp


def shap_analysis(model: Pipeline, X: pd.DataFrame, scenario: str,
                  cfg: RunConfig) -> Optional[pd.DataFrame]:
    """SHAP beeswarm + bar for the final model.

    Robust to the two SHAP-shape traps that broke the notebooks:
      * ``shap_values`` may be a list [class0, class1] OR a 3-D array
        (n, features, classes) OR a 2-D array, depending on version and model.
        Notebook 3 did ``sv = shap_values[1] if isinstance(list)`` and then
        ``sv[:, :, 1]`` — those two are mutually exclusive, so one path always
        threw or silently selected the wrong class.
      * The explained matrix must be the POST-preprocessing matrix with matching
        feature names; notebook 3 passed the raw ``X`` (still containing NaNs).
    """
    section(f"9.2  SHAP analysis ({scenario})")
    try:
        import shap
    except ImportError:
        print("  shap not installed — skipping")
        return None

    feat = model.named_steps["features"]
    clf = model.named_steps["model"]
    Xt = feat.transform(X)
    names = list(feat.named_steps["encode"].get_feature_names_out())
    Xt_df = pd.DataFrame(np.asarray(Xt), columns=names)
    assert not Xt_df.isna().any().any(), "NaNs survived preprocessing"
    print(f"  explained matrix : {Xt_df.shape[0]} x {Xt_df.shape[1]}")

    n_show = min(cfg.shap_max_samples, len(Xt_df))
    rs = np.random.RandomState(cfg.random_state)
    sample_idx = rs.choice(len(Xt_df), n_show, replace=False)
    X_exp = Xt_df.iloc[sample_idx]

    sv2d: Optional[np.ndarray] = None
    try:
        expl = shap.TreeExplainer(clf)
        raw = expl.shap_values(X_exp, check_additivity=False)
        if isinstance(raw, list):
            sv2d = np.asarray(raw[1] if len(raw) > 1 else raw[0])
        else:
            raw = np.asarray(raw)
            sv2d = raw[:, :, 1] if raw.ndim == 3 else raw
        print("  explainer: TreeExplainer (exact)")
    except Exception as exc:
        print(f"  TreeExplainer unavailable ({type(exc).__name__}); "
              f"falling back to PermutationExplainer")
        try:
            bg = shap.sample(Xt_df, min(50, len(Xt_df)), random_state=cfg.random_state)
            expl = shap.Explainer(clf.predict_proba, bg)
            ex = expl(X_exp, max_evals=800 if cfg.fast else 2000)
            vals = np.asarray(ex.values)
            sv2d = vals[:, :, 1] if vals.ndim == 3 else vals
        except Exception as exc2:
            print(f"  SHAP failed entirely: {type(exc2).__name__}: {exc2}")
            return None

    if sv2d is None or sv2d.ndim != 2:
        print("  unexpected SHAP shape — skipping plots")
        return None

    mean_abs = np.abs(sv2d).mean(axis=0)
    tab = (pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    save_table(tab, cfg.outdir, f"T31_shap_importance_{scenario}")
    print(tab.head(15).to_string(index=False))

    # --- beeswarm
    try:
        fig = plt.figure(figsize=(8.0, 6.5), dpi=140)
        shap.summary_plot(sv2d, X_exp, max_display=20, show=False)
        plt.title(f"SHAP — impact on predicted failure risk ({scenario})",
                  fontsize=11)
        savefig(plt.gcf(), cfg.outdir, f"F31_shap_beeswarm_{scenario}")
    except Exception as exc:
        print(f"  beeswarm failed: {exc}")
        plt.close("all")

    # --- bar
    try:
        fig = plt.figure(figsize=(8.0, 6.5), dpi=140)
        shap.summary_plot(sv2d, X_exp, plot_type="bar", max_display=20, show=False)
        plt.title(f"SHAP — mean |impact| ({scenario})", fontsize=11)
        savefig(plt.gcf(), cfg.outdir, f"F32_shap_bar_{scenario}")
    except Exception as exc:
        print(f"  bar plot failed: {exc}")
        plt.close("all")

    return tab


# =============================================================================
# 9.3  LINEAR MODEL COEFFICIENTS  +  9.4 SINGLE-PREDICTOR BENCHMARK
# =============================================================================

def linear_coefficient_table(model: Pipeline, scenario: str,
                             cfg: RunConfig) -> Optional[pd.DataFrame]:
    """Standardised coefficients and odds ratios for a linear final model.

    When the winning model is a penalised logistic regression, the coefficients
    ARE the explanation — far more useful to a clinical reader than a SHAP
    beeswarm, and they show directly how many predictors survived the L1
    penalty. Neither notebook reported them (no linear model was fitted in
    notebook 3 at all).
    """
    section(f"9.3  Coefficients of the final linear model ({scenario})")
    clf = model.named_steps["model"]
    if not hasattr(clf, "coef_"):
        print("  final model is not linear — skipping coefficient table")
        return None

    names = list(model.named_steps["features"].named_steps["encode"]
                 .get_feature_names_out())
    coef = np.asarray(clf.coef_).ravel()
    tab = pd.DataFrame({
        "feature": names,
        "coefficient": coef,
        "odds_ratio": np.exp(coef),
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    n_nonzero = int((tab["coefficient"].abs() > 1e-10).sum())
    print(f"  predictors entering the model : {n_nonzero} of {len(tab)} "
          f"({n_nonzero / len(tab) * 100:.0f}%)")
    print(f"  intercept                     : {float(clf.intercept_[0]):.4f}")
    print("\n  Non-zero coefficients (features are standardised, so these are "
          "directly comparable):")
    print(tab[tab["abs_coefficient"] > 1e-10]
          .drop(columns="abs_coefficient").round(4).to_string(index=False))

    save_table(tab.drop(columns="abs_coefficient"), cfg.outdir,
               f"T32_linear_coefficients_{scenario}")

    nz = tab[tab["abs_coefficient"] > 1e-10].head(20).iloc[::-1]
    if len(nz):
        fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(nz) + 1.8), dpi=140)
        colors = ["#c0392b" if c > 0 else "#2471a3" for c in nz["coefficient"]]
        ax.barh(nz["feature"], nz["coefficient"], color=colors)
        ax.axvline(0, color="0.2", lw=1)
        ax.set_xlabel("Standardised log-odds coefficient\n"
                      "(red = increases risk of dissatisfaction, blue = protective)")
        ax.set_title(f"Final model coefficients — {scenario}")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        savefig(fig, cfg.outdir, f"F33_linear_coefficients_{scenario}")
    return tab


def single_predictor_benchmark(X: pd.DataFrame, y: np.ndarray,
                               oof: Dict[str, np.ndarray], best: str,
                               scenario: str, cfg: RunConfig
                               ) -> Optional[pd.DataFrame]:
    """Does the ML model beat its single strongest predictor?

    This is the question that decides whether a machine-learning paper on this
    dataset has a result at all. If a univariable logistic regression on one
    variable matches the tuned multivariable ensemble, the added complexity buys
    nothing and should not be presented as if it did. Neither notebook ran this
    comparison.
    """
    section(f"9.4  Single-predictor benchmark ({scenario})")
    candidates = [c for c in ["satisfactionV3", "satisfactionV2", "pain",
                              "thicknessV0"] if c in X.columns]
    if not candidates:
        print("  no candidate single predictors in this scenario — skipping")
        return None

    outer_cv = StratifiedKFold(cfg.outer_splits, shuffle=True,
                               random_state=cfg.random_state)
    rows: List[Dict[str, Any]] = []
    uni_proba: Dict[str, np.ndarray] = {}

    for col in candidates:
        Xi = X[[col]].copy()
        pipe = Pipeline([
            ("features", make_feature_pipeline(cfg.random_state, scale=True,
                                               add_deltas=False,
                                               mice_iter=5 if cfg.fast else 10)),
            ("model", LogisticRegression(class_weight="balanced",
                                         max_iter=2000,
                                         random_state=cfg.random_state)),
        ])
        try:
            p = cross_val_predict(pipe, Xi, y, cv=outer_cv,
                                  method="predict_proba", n_jobs=1)[:, 1]
        except Exception as exc:
            print(f"  [skip] {col}: {exc}")
            continue
        uni_proba[col] = p
        pt, lo, hi = bootstrap_auc_ci(y, p, cfg.n_bootstrap,
                                      random_state=cfg.random_state)
        rows.append({"predictor": f"{col} alone (univariable LR)",
                     "AUC": round(pt, 3), "CI95_low": round(lo, 3),
                     "CI95_high": round(hi, 3)})
        print(f"  {col:<18} alone : AUC = {pt:.3f} [{lo:.3f}–{hi:.3f}]")

    mats = [oof[best][m] for m in range(oof[best].shape[0])
            if not np.isnan(oof[best][m]).any()]
    p_full = np.mean(np.vstack(mats), axis=0)
    pt, lo, hi = bootstrap_auc_ci(y, p_full, cfg.n_bootstrap,
                                  random_state=cfg.random_state)
    rows.append({"predictor": f"{best} (all {X.shape[1]} predictors)",
                 "AUC": round(pt, 3), "CI95_low": round(lo, 3),
                 "CI95_high": round(hi, 3)})
    print(f"  {'FULL MODEL':<18}       : AUC = {pt:.3f} [{lo:.3f}–{hi:.3f}]")

    for col, p in uni_proba.items():
        r = delong_like_bootstrap_test(y, p_full, p, n_boot=cfg.n_bootstrap,
                                       random_state=cfg.random_state)
        rows.append({"predictor": f"→ full model MINUS {col} alone",
                     "AUC": round(r["auc_diff"], 3),
                     "CI95_low": round(r["ci_low"], 3),
                     "CI95_high": round(r["ci_high"], 3),
                     "p_value": round(r["p_value"], 4)})
        verdict = ("full model is better"
                   if r["p_value"] < 0.05 and r["auc_diff"] > 0
                   else "NO significant gain over the single predictor")
        print(f"  full model vs {col} alone: ΔAUC = {r['auc_diff']:+.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}], "
              f"p = {r['p_value']:.3f}  →  {verdict}")

    tab = pd.DataFrame(rows)
    save_table(tab, cfg.outdir, f"T33_single_predictor_benchmark_{scenario}")

    plot_rows = tab[~tab["predictor"].str.startswith("→")]
    fig, ax = plt.subplots(figsize=(8.0, 0.62 * len(plot_rows) + 1.8), dpi=140)
    ypos = np.arange(len(plot_rows))
    colors = ["#1f4e79" if "all " in s else "#8fb8d8"
              for s in plot_rows["predictor"]]
    ax.barh(ypos, plot_rows["AUC"], color=colors, height=0.6)
    ax.errorbar(plot_rows["AUC"], ypos,
                xerr=[plot_rows["AUC"] - plot_rows["CI95_low"],
                      plot_rows["CI95_high"] - plot_rows["AUC"]],
                fmt="none", ecolor="0.25", capsize=4, lw=1.2)
    ax.axvline(0.5, ls=":", color="0.45", lw=1.4)
    ax.set_yticks(ypos); ax.set_yticklabels(plot_rows["predictor"])
    ax.invert_yaxis(); ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("Cross-validated AUC (95% bootstrap CI)")
    ax.set_title(f"Does the full model beat its strongest single predictor?\n"
                 f"scenario {scenario}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, cfg.outdir, f"F34_single_predictor_benchmark_{scenario}")
    return tab


# =============================================================================
# 10. SENSITIVITY ANALYSES
# =============================================================================

def threshold_sensitivity(df: pd.DataFrame, X_dict: Dict[str, pd.DataFrame],
                          cfg: RunConfig) -> pd.DataFrame:
    """Does the conclusion depend on where 'failure' is cut?

    The 7/10 cut-off is a convention, not a measurement. If AUC collapses at 6
    or 8, the model is really tracking one arbitrary bin boundary rather than
    dissatisfaction.
    """
    banner("10. SENSITIVITY ANALYSIS — outcome threshold")
    rows: List[Dict[str, Any]] = []
    inner_cv = StratifiedKFold(cfg.inner_splits, shuffle=True,
                               random_state=cfg.random_state)
    outer_cv = StratifiedKFold(cfg.outer_splits, shuffle=True,
                               random_state=cfg.random_state)
    specs = {s.name: s for s in build_model_specs(cfg.random_state, fast=cfg.fast)}
    probe_names = [n for n in ["Logistic Regression (L2)", "Random Forest", "XGBoost"]
                   if n in specs]

    for thr in SENSITIVITY_THRESHOLDS:
        y_thr = (df[TARGET_CONT].astype(float) < thr).astype(int).values
        rate = float(y_thr.mean())
        for scen, X in X_dict.items():
            add_deltas = scen.startswith("B")
            for nm in probe_names:
                spec = specs[nm]
                feat = make_feature_pipeline(cfg.random_state, spec.needs_scaling,
                                             add_deltas,
                                             mice_iter=5 if cfg.fast else 8)
                search = _make_search(spec, feat, inner_cv, cfg)
                try:
                    p = cross_val_predict(search, X, y_thr, cv=outer_cv,
                                          method="predict_proba", n_jobs=1)[:, 1]
                    a = roc_auc_score(y_thr, p)
                except Exception as exc:
                    print(f"  [skip] thr={thr} {scen} {nm}: {exc}")
                    continue
                rows.append({"threshold": thr, "failure_rate": round(rate, 3),
                             "n_failures": int(y_thr.sum()),
                             "scenario": scen, "model": nm,
                             "AUC": round(float(a), 3)})
                print(f"  threshold<{thr:g}  {scen:<12} {nm:<26} "
                      f"AUC={a:.3f}  (events={int(y_thr.sum())})", flush=True)

    tab = pd.DataFrame(rows)
    save_table(tab, cfg.outdir, "T40_threshold_sensitivity")

    if not tab.empty:
        fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=140)
        for (scen, nm), g in tab.groupby(["scenario", "model"]):
            ls = "-" if scen.startswith("A") else "--"
            ax.plot(g["threshold"], g["AUC"], marker="o", ls=ls,
                    label=f"{scen} · {nm}")
        ax.axhline(0.5, ls=":", color="0.5")
        ax.set_xlabel("Failure threshold on satisfactionV4 (< threshold = failure)")
        ax.set_ylabel("Cross-validated AUC")
        ax.set_title("Sensitivity of discrimination to the outcome cut-off")
        ax.set_ylim(0.4, 1.0); ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        savefig(fig, cfg.outdir, "F40_threshold_sensitivity")
    return tab


def scenario_comparison_figure(results: Dict[str, pd.DataFrame],
                               cfg: RunConfig) -> pd.DataFrame:
    """Side-by-side AUC of every classifier in scenario A vs scenario B."""
    banner("11. SCENARIO COMPARISON (baseline vs. baseline+follow-up)")
    frames = []
    for scen, summ in results.items():
        frames.append(summ[["Classifier", "_auc_mean", "_pooled_auc_mean"]]
                      .assign(scenario=scen))
    tab = pd.concat(frames, ignore_index=True)
    wide = tab.pivot(index="Classifier", columns="scenario",
                     values="_auc_mean").round(3)
    wide["difference (B - A)"] = (wide.get("B_full", np.nan)
                                  - wide.get("A_baseline", np.nan)).round(3)
    wide = wide.sort_values("difference (B - A)", ascending=False)
    print(wide.to_string())
    save_table(wide.reset_index(), cfg.outdir, "T41_scenario_comparison")

    labels = wide.index.tolist()
    ypos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(labels) + 2.2), dpi=140)
    if "A_baseline" in wide:
        ax.barh(ypos - 0.2, wide["A_baseline"], height=0.4,
                color="#7fb3d5", label="A — baseline (V0) only")
    if "B_full" in wide:
        ax.barh(ypos + 0.2, wide["B_full"], height=0.4,
                color="#1f4e79", label="B — baseline + V2 + V3")
    ax.axvline(0.5, ls=":", color="0.4", lw=1.2)
    ax.set_yticks(ypos); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlim(0.3, 1.0); ax.set_xlabel("Nested cross-validated AUC")
    ax.set_title("Discrimination by predictor scenario")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, cfg.outdir, "F41_scenario_comparison")
    return wide


def export_excel_workbook(outdir: Path) -> Optional[Path]:
    """Bundle every CSV table into one multi-sheet workbook."""
    section("12.1  Excel workbook")
    tabdir = outdir / "tables"
    csvs = sorted(tabdir.glob("*.csv"))
    if not csvs:
        return None
    xlsx = outdir / "HA_filler_all_results.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
            for f in csvs:
                sheet = f.stem[:31]
                try:
                    pd.read_csv(f).to_excel(xw, sheet_name=sheet, index=False)
                except Exception as exc:
                    print(f"  [skip] {f.name}: {exc}")
        print(f"  workbook saved -> {xlsx.name} ({len(csvs)} sheets)")
        return xlsx
    except Exception as exc:
        print(f"  could not write workbook: {exc}")
        return None


# =============================================================================
# 13. MAIN
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Leakage-safe ML pipeline for HA-filler satisfaction data")
    ap.add_argument("--data", type=Path, default=Path("HA_filler.sav"),
                    help="path to the SPSS .sav file")
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--fast", action="store_true",
                    help="smoke test: 2 imputations, 3 models, tiny search")
    ap.add_argument("--imputations", type=int, default=None)
    ap.add_argument("--n-iter", type=int, default=None)
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args(argv)

    cfg = RunConfig(data_path=args.data, outdir=args.outdir, fast=args.fast,
                    random_state=args.seed)
    if args.imputations:
        cfg.n_imputations = args.imputations
    if args.n_iter:
        cfg.n_iter_search = args.n_iter

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    (cfg.outdir / "tables").mkdir(exist_ok=True)
    (cfg.outdir / "figures").mkdir(exist_ok=True)

    tee = Tee(cfg.outdir / "analysis_log.txt")
    sys.stdout = tee
    t_start = time.time()

    try:
        env = print_env(cfg)
        np.random.seed(cfg.random_state)

        # ---------------------------------------------------- 1. data
        raw = load_data(cfg.data_path)
        section("1.2  Missingness")
        miss = missing_report(raw)
        print(miss[miss["missing_n"] > 0].to_string(index=False))
        save_table(miss, cfg.outdir, "T00_missingness")

        dropout_analysis(raw, cfg.outdir)
        df = build_outcome(raw, FAILURE_THRESHOLD)
        y = df[OUTCOME_BIN].values.astype(int)

        # ---------------------------------------------------- 2. descriptives
        make_tableone(df, cfg.outdir)

        # ---------------------------------------------------- 3. scenarios
        X_dict = split_predictors(df)

        # -------------------------------------------- 4-9. per scenario
        summaries: Dict[str, pd.DataFrame] = {}
        all_oof: Dict[str, Dict[str, np.ndarray]] = {}
        for scen, X in X_dict.items():
            add_deltas = scen.startswith("B")
            with Timer(f"nested CV — {scen}"):
                summary, oof, _ = nested_cv_benchmark(X, y, scen, cfg, add_deltas)
            summaries[scen] = summary
            all_oof[scen] = oof

            plot_roc_all_models(y, oof, scen, cfg)
            plot_pr_curves(y, oof, scen, cfg)
            plot_metric_comparison(summary, scen, cfg)

            best_name = summary.iloc[0]["Classifier"]
            print(f"\n  best classifier by mean AUC in {scen}: {best_name}")
            plot_calibration(y, oof, best_name, scen, cfg)
            top3 = summary["Classifier"].head(3).tolist()
            plot_decision_curve(y, oof, top3, scen, cfg)
            plot_confusion(y, oof, best_name, scen, cfg)
            pairwise_auc_tests(y, oof, scen, cfg)

            spec = {s.name: s for s in build_model_specs(cfg.random_state,
                                                         fast=cfg.fast)}[best_name]
            with Timer(f"final model + explainability — {scen}"):
                final = fit_final_model(X, y, spec, cfg, add_deltas)
                permutation_importance_plot(final, X, y, scen, cfg)
                linear_coefficient_table(final, scen, cfg)
                shap_analysis(final, X, scen, cfg)
                single_predictor_benchmark(X, y, oof, best_name, scen, cfg)

        # ---------------------------------------------------- 10-11
        scenario_comparison_figure(summaries, cfg)
        if not args.skip_sensitivity:
            threshold_sensitivity(df, X_dict, cfg)

        # ---------------------------------------------------- 12. exports
        banner("12. EXPORTS")
        export_excel_workbook(cfg.outdir)
        with open(cfg.outdir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"environment": env,
                       "config": {k: str(v) for k, v in cfg.__dict__.items()},
                       "n_analysed": int(len(df)),
                       "n_failures": int(y.sum()),
                       "failure_rate": float(y.mean()),
                       "runtime_seconds": round(time.time() - t_start, 1)},
                      f, indent=2)
        print(f"  metadata saved -> run_metadata.json")

        banner("DONE")
        print(f"  total runtime : {time.time() - t_start:,.1f}s")
        print(f"  outputs in    : {cfg.outdir.resolve()}")
        return 0

    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
