# World Happiness — Data Mining Pipeline (Java + Weka)

**Short:** a small Java + Weka pipeline that cleans the World Happiness Report CSV, creates a 3-class target (`happiness_level`), trains & evaluates two classification models (RandomForest and J48 with an appended KMeans cluster attribute), shows visualizations (ROC, cluster plot), and saves cleaned ARFF outputs.

---

## Table of contents

- [What this repo contains](#what-this-repo-contains)  
- [Requirements](#requirements)  
- [How to run](#how-to-run)  
- [Project structure](#project-structure)  
- [Pipeline steps (high-level)](#pipeline-steps-high-level)  
- [Algorithms: purpose & why used](#algorithms-purpose--why-used)  
- [Evaluation methodology & results summary](#evaluation-methodology--results-summary)  
- [Troubleshooting & tips](#troubleshooting--tips)  
- [Next steps & contact](#next-steps--contact)  
- [License](#license)

---

## What this repo contains

- `src/Main.java` — main pipeline (preprocessing, modelling, evaluation, visualization).  
- `bin/` — compiled classes (optional).  
- `data/`:
  - `World Happiness Report 2024.csv` (input CSV)
  - `world_happiness_cleaned.arff` (saved cleaned dataset)
  - `world_happiness_latest_by_country.arff` (one-row-per-country snapshot)
- `lib/` — required jars (e.g. `weka.jar`).  
- `run.bat` — convenience script to run the project on Windows.  
- `Project_Overview.docx` — extended explanation of choices and steps.

---

## Requirements

- Java JDK 11+ (or later).  
- Weka JAR(s) present in the `lib/` folder (the code uses Weka APIs).  
- Swing GUI available if you want visualizations (`ENABLE_VIS` default: true). On headless servers set `ENABLE_VIS = false` in `Main.java`.

---

## How to run

**Windows (simple):**

1. Open PowerShell or CMD at repository root.
2. Run:
```powershell
.\run.bat
```

**Manual (cross-platform):**

1. Compile:
```bash
javac -cp "lib/*" -d bin src/*.java
```
2. Run:
```bash
# Windows
java -cp "bin;lib/*" src.Main

# macOS / Linux
java -cp "bin:lib/*" src.Main
```

> If you get classpath errors, ensure `lib/weka.jar` exists and `-cp` includes `lib/*`.

---

## Project structure (quick)

```
/ (repo root)
├─ bin/
├─ data/
│  ├─ World Happiness Report 2024.csv
│  ├─ world_happiness_cleaned.arff
│  └─ world_happiness_latest_by_country.arff
├─ lib/
│  └─ weka.jar (and other jars)
├─ src/
│  └─ Main.java
├─ run.bat
└─ Project_Overview.docx
```

---

## Pipeline steps (high-level, what the code does)

1. **Load CSV** using Weka `CSVLoader`.  
2. **Preprocessing**
   - Remove totally empty attributes.
   - Remove exact duplicate rows.
   - Winsorize numeric outliers using the IQR (1.5×IQR) rule.
   - Replace missing values (numeric → mean, nominal → mode).
   - Detect `Life Ladder` numeric column, convert it to a nominal 3-class target `happiness_level` (Low/Medium/High) using 33% and 66% quantiles, then remove the numeric column.
   - Save cleaned ARFF (`data/world_happiness_cleaned.arff`) and a "latest-per-country" ARFF file.
3. **Model A — RandomForest**
   - Standardize numeric attributes (safe copy).
   - 10-fold cross-validation (CV) for evaluation; print accuracy, kappa, RMSE, MAE, per-class metrics and confusion matrix.
   - Train RandomForest on full dataset for sample predictions and ROC visualization.
   - Optionally launch ROC visualization (one curve per class).
4. **Model B — J48 + AddCluster**
   - Append a nominal `cluster` attribute using `AddCluster` with `SimpleKMeans` (k = `CLUSTER_K`).
   - Visualize clusters (colored plot).
   - 10-fold CV using J48 (with cluster attribute present), report metrics.
   - Train final J48 on the full dataset and print the textual pruned decision tree.
5. **Comparison**
   - Print a compact comparison summary (Accuracy, Kappa, CV time) and refer to per-class details printed earlier.

---

## Algorithms — purpose & why these were chosen

**Purpose of classification algorithms (simple):** predict a categorical label (here: `Low`, `Medium`, `High`) from features.

**Why RandomForest?**
- Strong performance on tabular data.
- Handles numeric/nominal features without heavy tuning.
- Robust to overfitting due to ensemble averaging.
- Good baseline for predictive accuracy and ROC AUC.

**Why J48 (C4.5) + AddCluster?**
- J48 produces an interpretable decision tree — easy to inspect rules.
- `AddCluster` inserts unsupervised cluster IDs as a predictor so the tree can use latent group structure (helpful for group-specific rules).
- Use J48+cluster to evaluate whether cluster-derived features help and to provide interpretable insights.

**Design rationale:** one model (RandomForest) for predictive strength, the other (J48 + cluster) for interpretability and exploratory feature engineering.

---

## Evaluation methodology & results summary

- Both models are evaluated using **10-fold cross-validation** (`CV_FOLDS = 10`) with a fixed random seed for reproducibility.
- Metrics printed:
  - Overall accuracy, Kappa.
  - MAE, RMSE, relative errors.
  - Per-class precision, recall, F1, ROC AUC.
  - Confusion matrix.
  - Sample predictions and distributions.
- Example (from a run):
  - RandomForest: `Accuracy ≈ 0.84`, `Kappa ≈ 0.76`.
  - J48 + AddCluster: `Accuracy ≈ 0.78`, `Kappa ≈ 0.675`.
- **Remarks:** RandomForest gave better predictive performance in the example run; J48+cluster provided an interpretable tree with country- and cluster-specific splits which can help explain model behavior.

---

## Troubleshooting & tips

- **Compilation error:** `local variables referenced from a lambda must be final` — make variables used inside `EventQueue.invokeLater(...)` effectively final (don't reassign).
- **Headless server / no GUI**: set `ENABLE_VIS = false` in `Main.java` to skip visualizations.
- **Reproducibility**: keep `RANDOM_SEED` fixed for CV splits and clustering.
- **Run on latest-per-country dataset:** replace `data` with `latestPerCountry` before training if you want one-row-per-country evaluation.

---

## Next steps (suggestions)

- Add repeated CV and statistical tests (paired t-test / McNemar) to assess significance of accuracy differences.
- Save per-fold metrics to CSV for plotting and analysis.
- Export ROC plots and confusion matrix images to `results/` for report inclusion.
- Try additional models (e.g., XGBoost via Java wrapper, SVM, logistic regression) and hyperparameter tuning.

---

## License

Provided as-is. Add a license file (e.g., MIT) if you want to explicitly permit reuse.

---

If you'd like, I can also:
- Add the exact numeric results from your latest run into a `RESULTS.md` and include generated ROC PNGs; or
- Add a small script to export CV fold results to CSV for statistical tests.
