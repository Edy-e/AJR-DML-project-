# Does Double/Debiased Machine Learning Change the AJR Estimate?
### A Coefficient-Comparison Test in a Small-Sample IV Design

**Wenjing Gao & Edith Jiang**
EC610I — Machine Learning in Economics, Wilfrid Laurier University
July 2026

---

## Overview

This project extends Acemoglu, Johnson, and Robinson's (2001, AJR) instrumental-variables
analysis of institutions and long-run GDP by comparing classical 2SLS to Double/Debiased
Machine Learning (DML; Chernozhukov et al., 2018) across three nested control
specifications and five DML nuisance learners (Lasso, Ridge, Elastic Net, Random Forest,
Gradient Boosting).

Rather than asking whether "machine learning improves causal inference" in the abstract,
we ask a single, directly testable question: **does replacing the linear nuisance
functions inside AJR's IV estimator with machine learning produce a coefficient that is
statistically different from the original 2SLS estimate?** For every DML estimate we
report a formal difference-in-coefficients test against its same-specification 2SLS
counterpart, alongside the change in standard error and confidence-interval overlap.

## Key Result

Across all 15 DML-vs-2SLS comparisons (3 specifications × 5 learners), **no DML
coefficient is statistically distinguishable from 2SLS at the 5% level**. However,
the three regularized linear learners (Lasso, Ridge, Elastic Net) achieve standard
errors **25–45% smaller** than 2SLS in the specifications where the classical first
stage is weakest, while landing on essentially the same coefficient. Tree-ensemble
learners (Random Forest, Gradient Boosting) shift point estimates substantially — in
one specification, more than doubling the coefficient — but their own standard errors
inflate so much that the shift carries no statistical weight, consistent with
small-sample estimator instability rather than a competing causal finding.

## Repository Structure

```
AJR-DML-project/
├── README.md
├── report/
│   ├── AJR_ML_final_report.tex          Final report (LaTeX source)
│   ├── AJR_ML_final_report.pdf          Final report (compiled PDF)
│   ├── main_coefficient_comparison.png  Main figure (required to compile)
│   └── se_comparison.png                Appendix figure (required to compile)
├── code/
│   └── AJR_ml_extension_final.py        Single reproducible analysis script
├── data/
│   └── AJR_final_dataset.csv            AJR (2012) replication dataset, N = 64
└── output/
    ├── tables/
    │   ├── main_coefficient_table.csv
    │   └── coefficient_equality_tests.csv
    ├── figures/
    │   ├── main_coefficient_comparison.png / .pdf
    │   └── se_comparison.png / .pdf
    └── run_manifest.json                 Reproducibility record (SHA-256, library versions)
```

## Data

`AJR_final_dataset.csv` is the AJR (2012) reply dataset: 64 former European colonies.

| Variable | Description |
|---|---|
| `loggdp` | log GDP per capita (outcome) |
| `risk` | protection against expropriation risk, ICRG 1985–95 (endogenous regressor) |
| `logmort0` | log settler mortality (instrument) |
| `latitude`, `malfal94`, `edes1975` | AJR-motivated robustness controls |
| `log_pop_density`, `log_trade`, `fdi_pct_gdp`, `life_expectancy` | possible mediators, used only in the Spec-3 sensitivity check |
| `hci`, `govt_effectiveness` | post-treatment variables, not used in the primary specifications |

## How to Reproduce

```bash
cd code
python AJR_ml_extension_final.py
```

**Requirements:** Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy`.

```bash
pip install pandas numpy scikit-learn matplotlib scipy
```

The script automatically locates `AJR_final_dataset.csv` whether it sits in the same
folder as the script, in a `data/` subfolder, or in a sibling `data/` folder (as in
this repository's layout) — no path editing is required regardless of where the
repository is cloned.

Running the script:
1. Reproduces AJR's no-controls 2SLS baseline exactly.
2. Estimates 2SLS and five DML learners under three specifications (18 estimates total).
3. Runs a formal coefficient-equality test between every DML estimate and its
   same-specification 2SLS counterpart (15 tests).
4. Writes all tables (`output/tables/`), figures (`output/figures/`), and a
   reproducibility manifest (`output/run_manifest.json`) recording the dataset's
   SHA-256 hash and the exact library versions used.

Total runtime: under one minute on a standard laptop.

## How to Compile the Report

```bash
cd report
pdflatex AJR_ML_final_report.tex
pdflatex AJR_ML_final_report.tex
```

(Run `pdflatex` twice so that table/figure cross-references resolve correctly.)

## Method Summary

For each specification, we estimate the causal effect two ways:

- **2SLS**, with HC1-robust standard errors, exactly as in AJR's own approach.
- **DML**, in the partially linear IV model of Chernozhukov et al. (2018), with
  $K=5$-fold cross-fitting and five nuisance learners.

For every DML estimate we compute an approximate difference-in-coefficients statistic
against the same-specification 2SLS estimate:

$$z = \frac{\hat\beta_1^{\text{DML}} - \hat\beta_1^{\text{2SLS}}}{\sqrt{(\text{SE}^{\text{DML}})^2 + (\text{SE}^{\text{2SLS}})^2}}$$

This statistic is analogous in spirit to a Hausman-style comparison but is not a
formal Hausman test, since it does not estimate the covariance between the two
estimators. Full methodological detail, including limitations of this approximation,
is in Section 3 of the report.

## Limitations

See Section 6 of the report for full discussion. In brief: (i) the approximate
equality test does not estimate the covariance between 2SLS and DML, since both are
computed on the same sample; (ii) at N = 56–64, asymptotic justifications for both
the DML standard error and the equality test warrant some caution; (iii) the first
stage is weak in Specifications 2–3 (F below the conventional threshold of 10),
so point estimates from either estimator in those specifications should be read
with appropriate caution.

## References

- Acemoglu, D., Johnson, S., Robinson, J.A. (2001). The Colonial Origins of
  Comparative Development. *American Economic Review*, 91(5), 1369–1401.
- Acemoglu, D., Johnson, S., Robinson, J.A. (2012). The Colonial Origins of
  Comparative Development: Reply. *American Economic Review*, 102(6), 3077–3110.
- Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for Treatment
  and Structural Parameters. *Econometrics Journal*, 21(1), C1–C68.
- Curtin, P.D. (1989). *Death by Migration*. Cambridge University Press.
- Hausman, J.A. (1978). Specification Tests in Econometrics. *Econometrica*,
  46(6), 1251–1271.
