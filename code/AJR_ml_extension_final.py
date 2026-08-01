"""
=============================================================================
Double/Debiased Machine Learning Extension of Acemoglu, Johnson, and
Robinson (2001) -- "The Colonial Origins of Comparative Development"

SIMPLIFIED VERSION -- coefficient-comparison focused
(per instructor guidance: no predictive benchmark; focus entirely on
whether DML coefficients differ significantly from 2SLS, with formal
equality tests, SE comparison, and CI overlap -- mirroring the structure
of the Autor-Dorn-Hanson DML replication poster shown in class.)

Authors:     Wenjing Gao, Edith Jiang
Course:      EC610I -- Machine Learning in Economics
Instructor:  M. Jahangir Alam, PhD
Institution: Wilfrid Laurier University

WHAT THIS SCRIPT DOES
----------------------
1. Reproduces AJR's baseline 2SLS exactly.
2. Estimates the same causal effect across THREE control specifications
   (latitude only / AJR-robustness / AJR + mediators) using both 2SLS and
   five DML nuisance learners (Lasso, Ridge, Elastic Net, Random Forest,
   Gradient Boosting) -- a "specification x estimator" grid, exactly like
   the reference poster's Table-3-across-six-columns design.
3. For every DML estimate, runs a FORMAL coefficient-equality test against
   the 2SLS estimate from the SAME specification:
       z = (beta_DML - beta_2SLS) / sqrt(SE_DML^2 + SE_2SLS^2)
   with a two-sided p-value, plus an explicit check of whether the two
   95% confidence intervals overlap.
4. Produces ONE main comparison table (CSV) and ONE main coefficient plot
   (grouped error-bar chart, specification x estimator) as the centerpiece
   of the paper, mirroring the reference poster's main figure.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-------------------------------------------
No out-of-sample predictive benchmark, no Monte Carlo simulation, no PCA
post-treatment sensitivity, no Lasso subsample-selection analysis, no
tree-hyperparameter sensitivity grid. All of that was cut per instructor
guidance to keep the deliverable focused on one clear question: are the
DML coefficients statistically different from 2SLS, and by how much does
DML change precision?

Required packages: pandas, numpy, scikit-learn, matplotlib, scipy.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output_ml_simplified"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
for directory in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR):
    directory.mkdir(exist_ok=True, parents=True)

INNER_CV = 5  # inner CV folds inside LassoCV/RidgeCV/ElasticNetCV


# =============================================================================
# VARIABLE CLASSIFICATION -- three nested specifications ("columns")
# =============================================================================

LATITUDE_ONLY = ["latitude"]
AJR_ROBUSTNESS_CONTROLS = ["latitude", "malfal94", "edes1975"]
MEDIATORS = ["log_pop_density", "log_trade", "fdi_pct_gdp", "life_expectancy"]
AJR_PLUS_MEDIATORS = AJR_ROBUSTNESS_CONTROLS + MEDIATORS

# The three specifications we estimate with EVERY method, mirroring the
# reference poster's multi-column design (their Columns 1-6 = increasing
# control sets; ours = three increasingly demanding control sets).
SPECIFICATIONS: dict[str, list[str]] = {
    "Spec 1: Latitude only": LATITUDE_ONLY,
    "Spec 2: AJR robustness controls": AJR_ROBUSTNESS_CONTROLS,
    "Spec 3: AJR + mediators": AJR_PLUS_MEDIATORS,
}

CORE_REQUIRED_COLUMNS = {
    "loggdp", "risk", "logmort0",
    "latitude", "malfal94", "edes1975",
    "log_pop_density", "log_trade", "fdi_pct_gdp", "life_expectancy",
}


# =============================================================================
# DATA LOADING
# =============================================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_dataset() -> tuple[Path, pd.DataFrame]:
    # Search several relative locations so the script runs correctly
    # regardless of the repository layout: everything in one folder,
    # or a code/ + data/ sibling-folder structure, or a data/ subfolder
    # under the script. No absolute paths are used anywhere, so this
    # works on any machine that clones the repository.
    search_dirs = [
        SCRIPT_DIR,                      # same folder as this script
        SCRIPT_DIR / "data",             # ./data/ subfolder
        SCRIPT_DIR.parent / "data",      # ../data/ (e.g. code/ and data/ as siblings)
    ]
    filenames = [
        "AJR_final_dataset.csv",
        "AJR_final_dataset_v2.csv",
        "AJR_final_dataset(3).csv",
        "AJR_final_dataset(1).csv",
    ]
    candidates = [d / f for d in search_dirs for f in filenames]

    existing = []
    for path in candidates:
        if path.exists():
            try:
                cols = set(pd.read_csv(path, nrows=2).columns)
                existing.append((path, cols))
            except Exception as exc:
                print(f"WARNING: could not inspect {path}: {exc}")

    if not existing:
        searched = ", ".join(str(d) for d in search_dirs)
        raise FileNotFoundError(
            "No AJR dataset found. Searched: " + searched + ". "
            "Place AJR_final_dataset.csv in one of these folders."
        )

    for path, cols in existing:
        if CORE_REQUIRED_COLUMNS.issubset(cols):
            return path, pd.read_csv(path)

    best_path, best_cols = max(existing, key=lambda item: len(item[1] & CORE_REQUIRED_COLUMNS))
    missing = sorted(CORE_REQUIRED_COLUMNS - best_cols)
    raise ValueError(f"Dataset {best_path.name} is missing required columns: {', '.join(missing)}")


DATA_PATH, df = resolve_dataset()
print("=" * 78)
print(" DATA")
print("=" * 78)
print(f"Loaded {len(df)} countries, {df.shape[1]} columns, from {DATA_PATH.name}")
print(f"SHA-256: {sha256_file(DATA_PATH)[:16]}...")

missing_core = sorted(CORE_REQUIRED_COLUMNS - set(df.columns))
if missing_core:
    raise ValueError(f"Missing required columns: {', '.join(missing_core)}")

print("\nSpecifications to be estimated (2SLS + 5 DML learners, each):")
for name, controls in SPECIFICATIONS.items():
    print(f"  {name}: {controls}")


# =============================================================================
# CORE ESTIMATION FUNCTIONS
# =============================================================================

def complete_case(columns: Iterable[str], data: pd.DataFrame = df) -> pd.DataFrame:
    cols = list(dict.fromkeys(columns))
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return data[cols].dropna().copy()


def iv2sls(y: np.ndarray, endog: np.ndarray, exog: np.ndarray, iv: np.ndarray) -> dict[str, float]:
    """Just-identified 2SLS with conventional and HC1-robust standard errors."""
    y = np.asarray(y, dtype=float)
    endog = np.asarray(endog, dtype=float)
    exog = np.asarray(exog, dtype=float)
    iv = np.asarray(iv, dtype=float)
    n = len(y)
    Z = np.column_stack([exog, iv])

    gamma = np.linalg.lstsq(Z, endog, rcond=None)[0]
    endog_hat = Z @ gamma
    fs_resid = endog - endog_hat
    gamma_restricted = np.linalg.lstsq(exog, endog, rcond=None)[0]
    ssr_r = np.sum((endog - exog @ gamma_restricted) ** 2)
    ssr_u = np.sum(fs_resid ** 2)
    denom_df = n - Z.shape[1]
    first_stage_f = ((ssr_r - ssr_u) / 1.0) / (ssr_u / denom_df) if denom_df > 0 else np.nan

    X_hat = np.column_stack([exog, endog_hat])
    beta = np.linalg.lstsq(X_hat, y, rcond=None)[0]
    structural_X = np.column_stack([exog, endog])
    ss_resid = y - structural_X @ beta
    k = X_hat.shape[1]
    XtX_inv = np.linalg.pinv(X_hat.T @ X_hat)
    sigma2 = np.sum(ss_resid ** 2) / max(n - k, 1)
    se_conventional = np.sqrt(np.maximum(np.diag(sigma2 * XtX_inv), 0.0))
    meat = X_hat.T @ (X_hat * (ss_resid ** 2)[:, None])
    hc1_factor = n / max(n - k, 1)
    V_hc1 = XtX_inv @ meat @ XtX_inv * hc1_factor
    se_hc1 = np.sqrt(np.maximum(np.diag(V_hc1), 0.0))

    return {
        "beta": float(beta[-1]),
        "se_robust": float(se_hc1[-1]),
        "first_stage_f": float(first_stage_f),
        "n": int(n),
    }


def make_nuisance_pipeline(learner: Any) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("est", clone(learner)),
    ])


def fit_nuisance(learner: Any, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    pipe = make_nuisance_pipeline(learner)
    pipe.fit(X_train, y_train)
    return pipe


def dml_iv(
    y: np.ndarray, R: np.ndarray, M: np.ndarray, X: np.ndarray,
    learner: Any, n_folds: int = 5, seed: int = SEED,
) -> dict[str, Any]:
    """Cross-fitted partially-linear IV DML estimator (Chernozhukov et al.,
    2018). Same learner used for all three nuisance functions (y, R, M).
    SE is already heteroskedasticity-robust by construction (orthogonal
    score sandwich formula)."""
    y = np.asarray(y, dtype=float)
    R = np.asarray(R, dtype=float)
    M = np.asarray(M, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(y)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_tilde = np.zeros(n); R_tilde = np.zeros(n); M_tilde = np.zeros(n)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_tilde[test_idx] = y[test_idx] - fit_nuisance(learner, X_train, y[train_idx]).predict(X_test)
        R_tilde[test_idx] = R[test_idx] - fit_nuisance(learner, X_train, R[train_idx]).predict(X_test)
        M_tilde[test_idx] = M[test_idx] - fit_nuisance(learner, X_train, M[train_idx]).predict(X_test)

    denominator = float(np.sum(M_tilde * R_tilde))
    beta = float(np.sum(M_tilde * y_tilde) / denominator)
    psi = (y_tilde - beta * R_tilde) * M_tilde
    J = float(np.mean(M_tilde * R_tilde))
    se = float(np.sqrt(np.mean(psi ** 2) / (J ** 2 * n)))

    return {"beta": beta, "se_robust": se, "n": n}


LEARNERS = {
    "Lasso": LassoCV(cv=INNER_CV, random_state=SEED, max_iter=20_000, n_alphas=30),
    "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 30), cv=INNER_CV),
    "Elastic Net": ElasticNetCV(
        cv=INNER_CV, l1_ratio=[0.3, 0.5, 0.7, 0.9], random_state=SEED,
        max_iter=20_000, n_alphas=30,
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=SEED, n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150, max_depth=3, min_samples_leaf=5, learning_rate=0.05, random_state=SEED,
    ),
}


# =============================================================================
# COEFFICIENT EQUALITY TEST
# =============================================================================

def coefficient_equality_test(
    beta1: float, se1: float, beta2: float, se2: float,
) -> dict[str, float]:
    """Two-sample test of H0: beta1 = beta2, treating the two estimators'
    sampling distributions as independent (a standard, conservative way to
    compare coefficients from two different estimation procedures on the
    same data, in the absence of a tractable joint covariance). This is
    the same logic as a Hausman-style difference test:

        z = (beta1 - beta2) / sqrt(se1^2 + se2^2)

    Also reports whether the two individual 95% CIs overlap, which is a
    stricter (more conservative) criterion than the z-test itself: two
    CIs can fail to overlap only if the z-test already rejects at a much
    tighter level, so overlap/non-overlap is reported as a secondary,
    easy-to-visualize check alongside the formal p-value.
    """
    diff = beta1 - beta2
    se_diff = np.sqrt(se1 ** 2 + se2 ** 2)
    z = diff / se_diff if se_diff > 0 else np.nan
    p_value = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan

    ci1 = (beta1 - 1.96 * se1, beta1 + 1.96 * se1)
    ci2 = (beta2 - 1.96 * se2, beta2 + 1.96 * se2)
    overlap = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])

    return {
        "diff": float(diff),
        "se_diff": float(se_diff),
        "z_stat": float(z),
        "p_value": p_value,
        "significant_at_5pct": bool(p_value < 0.05) if np.isfinite(p_value) else False,
        "ci_overlap": bool(overlap),
    }


# =============================================================================
# MAIN ESTIMATION: SPECIFICATION x ESTIMATOR GRID
# =============================================================================

print("\n" + "=" * 78)
print(" MAIN ESTIMATION: 2SLS AND DML ACROSS THREE SPECIFICATIONS")
print("=" * 78)

# AJR's original no-controls baseline (reproduced once, used as reference
# throughout, and reported in every table for context).
baseline_data = complete_case(["loggdp", "risk", "logmort0"])
y_base = baseline_data["loggdp"].to_numpy(float)
R_base = baseline_data["risk"].to_numpy(float)
M_base = baseline_data["logmort0"].to_numpy(float)
BASELINE = iv2sls(y_base, R_base, np.ones((len(baseline_data), 1)), M_base)
print(
    f"\nAJR baseline (no controls): beta={BASELINE['beta']:.4f}, "
    f"HC1 SE={BASELINE['se_robust']:.4f}, F={BASELINE['first_stage_f']:.2f}, "
    f"N={BASELINE['n']}"
)
print("  (AJR 2012 Table 1B col 1 headline: beta=0.929, conventional SE=0.156)")

main_rows: list[dict[str, Any]] = []
equality_rows: list[dict[str, Any]] = []

for spec_name, controls in SPECIFICATIONS.items():
    data = complete_case(["loggdp", "risk", "logmort0"] + controls)
    y = data["loggdp"].to_numpy(float)
    R = data["risk"].to_numpy(float)
    M = data["logmort0"].to_numpy(float)
    X = data[controls].to_numpy(float)
    n = len(data)

    # 2SLS for this specification.
    tsls = iv2sls(y, R, np.column_stack([np.ones(n), X]), M)
    print(f"\n[{spec_name}] complete-case N={n}, controls={controls}")
    print(f"  2SLS:  beta={tsls['beta']:.4f}, HC1 SE={tsls['se_robust']:.4f}, F={tsls['first_stage_f']:.2f}")

    main_rows.append({
        "specification": spec_name, "n_controls": len(controls), "n": n,
        "estimator": "2SLS", "beta": tsls["beta"], "se_robust": tsls["se_robust"],
        "first_stage_f": tsls["first_stage_f"],
        "ci_lo": tsls["beta"] - 1.96 * tsls["se_robust"],
        "ci_hi": tsls["beta"] + 1.96 * tsls["se_robust"],
    })

    # DML for each of the five learners, same specification, same sample.
    for learner_name, learner in LEARNERS.items():
        result = dml_iv(y, R, M, X, learner, n_folds=5, seed=SEED)
        pct_change = 100.0 * (result["se_robust"] - tsls["se_robust"]) / tsls["se_robust"]
        tag = "SMALLER" if result["se_robust"] < tsls["se_robust"] else "LARGER"
        print(
            f"  DML {learner_name:18}: beta={result['beta']:8.4f}, "
            f"SE={result['se_robust']:.4f}  ({abs(pct_change):5.1f}% {tag} than 2SLS)"
        )

        main_rows.append({
            "specification": spec_name, "n_controls": len(controls), "n": n,
            "estimator": f"DML ({learner_name})", "beta": result["beta"],
            "se_robust": result["se_robust"], "first_stage_f": np.nan,
            "ci_lo": result["beta"] - 1.96 * result["se_robust"],
            "ci_hi": result["beta"] + 1.96 * result["se_robust"],
        })

        # Formal equality test: this DML estimate vs the 2SLS estimate from
        # THE SAME specification and THE SAME sample.
        test = coefficient_equality_test(
            result["beta"], result["se_robust"], tsls["beta"], tsls["se_robust"]
        )
        equality_rows.append({
            "specification": spec_name, "learner": learner_name,
            "beta_dml": result["beta"], "se_dml": result["se_robust"],
            "beta_2sls": tsls["beta"], "se_2sls": tsls["se_robust"],
            "se_pct_change": pct_change,
            **test,
        })

main_df = pd.DataFrame(main_rows)
equality_df = pd.DataFrame(equality_rows)
main_df.to_csv(TABLES_DIR / "main_coefficient_table.csv", index=False)
equality_df.to_csv(TABLES_DIR / "coefficient_equality_tests.csv", index=False)

print("\n" + "=" * 78)
print(" COEFFICIENT EQUALITY TESTS (DML vs 2SLS, same specification)")
print("=" * 78)
print(
    equality_df[[
        "specification", "learner", "beta_dml", "beta_2sls", "diff",
        "z_stat", "p_value", "significant_at_5pct", "ci_overlap", "se_pct_change",
    ]].round(4).to_string(index=False)
)

n_significant = equality_df["significant_at_5pct"].sum()
n_total = len(equality_df)
n_no_overlap = (~equality_df["ci_overlap"]).sum()
print(f"\nSummary: {n_significant}/{n_total} DML-vs-2SLS coefficient differences "
      f"are statistically significant at 5% ({n_significant/n_total:.0%}).")
print(f"         {n_no_overlap}/{n_total} pairs have non-overlapping 95% CIs "
      f"({n_no_overlap/n_total:.0%}).")


# =============================================================================
# MAIN FIGURE: COEFFICIENT COMPARISON PLOT
# =============================================================================
# Mirrors the reference-poster style: one horizontal panel per specification,
# one colored dot-and-whisker per estimator (2SLS + 5 DML learners), 95% CI
# error bars. This is intended to be THE central figure of the paper.

print("\n" + "=" * 78)
print(" MAIN FIGURE")
print("=" * 78)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

ESTIMATOR_ORDER = ["2SLS", "DML (Lasso)", "DML (Ridge)", "DML (Elastic Net)",
                   "DML (Random Forest)", "DML (Gradient Boosting)"]
ESTIMATOR_COLORS = {
    "2SLS": "black",
    "DML (Lasso)": "#1f77b4",
    "DML (Ridge)": "#2ca02c",
    "DML (Elastic Net)": "#d62728",
    "DML (Random Forest)": "#9467bd",
    "DML (Gradient Boosting)": "#ff7f0e",
}
ESTIMATOR_MARKERS = {
    "2SLS": "X",
    "DML (Lasso)": "o",
    "DML (Ridge)": "o",
    "DML (Elastic Net)": "o",
    "DML (Random Forest)": "o",
    "DML (Gradient Boosting)": "o",
}

spec_names = list(SPECIFICATIONS.keys())
fig, axes = plt.subplots(len(spec_names), 1, figsize=(8.5, 2.3 * len(spec_names)),
                          sharex=True)
if len(spec_names) == 1:
    axes = [axes]

for ax, spec_name in zip(axes, spec_names):
    subset = main_df[main_df["specification"] == spec_name]
    y_positions = np.arange(len(ESTIMATOR_ORDER))
    # Reference line = THIS specification's own 2SLS estimate, since that is
    # the actual comparison target in the equality tests (not the AJR
    # no-controls baseline, which is a different quantity).
    spec_2sls_beta = subset[subset["estimator"] == "2SLS"]["beta"].iloc[0]
    for i, estimator in enumerate(ESTIMATOR_ORDER):
        row = subset[subset["estimator"] == estimator]
        if row.empty:
            continue
        row = row.iloc[0]
        color = ESTIMATOR_COLORS[estimator]
        marker = ESTIMATOR_MARKERS[estimator]
        ax.errorbar(
            row["beta"], i, xerr=1.96 * row["se_robust"], fmt=marker, capsize=4,
            color=color, ecolor=color, markersize=9, markeredgecolor="black",
            markeredgewidth=0.6, linewidth=1.6,
        )
    ax.axvline(spec_2sls_beta, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ESTIMATOR_ORDER, fontsize=10)
    ax.set_title(spec_name, fontsize=11, loc="left", fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)

axes[-1].set_xlabel(r"Estimated Effect of Institutions on log GDP p.c. ($\hat\beta_1$, 95% CI)")
fig.suptitle(
    "Estimated Institutional Effect: 2SLS vs. Double Machine Learning\n"
    "(gray dashed line = that panel's own 2SLS estimate, the comparison used in the equality tests)",
    fontsize=12, y=1.00,
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "main_coefficient_comparison.png", dpi=200, bbox_inches="tight")
plt.savefig(FIGURES_DIR / "main_coefficient_comparison.pdf", bbox_inches="tight")
plt.close()
print("  main_coefficient_comparison.png/.pdf saved")


# =============================================================================
# SUPPLEMENTARY: SE COMPARISON BAR CHART
# =============================================================================
# A simple, direct visual of the "does DML reduce SE" question, organized
# by specification and estimator -- easy to read alongside the p-values.

fig, ax = plt.subplots(figsize=(8.5, 4.2))
width = 0.13
x = np.arange(len(spec_names))
for i, estimator in enumerate(ESTIMATOR_ORDER):
    ses = []
    for spec_name in spec_names:
        row = main_df[(main_df["specification"] == spec_name) & (main_df["estimator"] == estimator)]
        ses.append(row["se_robust"].iloc[0] if not row.empty else np.nan)
    offset = (i - len(ESTIMATOR_ORDER) / 2) * width + width / 2
    ax.bar(x + offset, ses, width=width, label=estimator, color=ESTIMATOR_COLORS[estimator],
           edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([s.replace("Spec ", "Spec\n") for s in spec_names], fontsize=9)
ax.set_ylabel("HC1-robust Standard Error")
ax.set_title("Standard Error by Specification and Estimator")
ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "se_comparison.png", dpi=200, bbox_inches="tight")
plt.savefig(FIGURES_DIR / "se_comparison.pdf", bbox_inches="tight")
plt.close()
print("  se_comparison.png/.pdf saved")


# =============================================================================
# REPRODUCIBILITY MANIFEST
# =============================================================================

manifest = {
    "script": Path(__file__).name,
    "data_file": str(DATA_PATH.resolve()),
    "data_file_name": DATA_PATH.name,
    "data_sha256": sha256_file(DATA_PATH),
    "rows": int(len(df)),
    "columns": int(df.shape[1]),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "seed": SEED,
    "specifications": {k: v for k, v in SPECIFICATIONS.items()},
    "baseline_result": BASELINE,
    "n_significant_at_5pct": int(n_significant),
    "n_total_tests": int(n_total),
    "n_ci_no_overlap": int(n_no_overlap),
}
with (OUTPUT_DIR / "run_manifest.json").open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, default=str)

print("\n" + "=" * 78)
print(" ALL DONE")
print(f"   Data:     {DATA_PATH}")
print(f"   Tables:   {TABLES_DIR}")
print(f"   Figures:  {FIGURES_DIR}")
print(f"   Manifest: {OUTPUT_DIR / 'run_manifest.json'}")
print("=" * 78)
