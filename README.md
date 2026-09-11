# Prediction of treatment failure after nasolabial hyaluronic acid filler

Analysis code for the manuscript *"Prediction of Non-Response to Hyaluronic Acid 
Injections for Improvement of Nasolabial Folds Using Longitudinal Clinical Features
and Machine Learning Models"*, under revision at **DARU Journal of Pharmaceutical Sciences**.

Ghadirzadeh M, Samadi A, Mohamadi F, Zahir-Jouzdani F.
Center for Research and Training in Skin Diseases and Leprosy, Tehran University
of Medical Sciences.

---

## Contents

| File | Description |
|---|---|
| `ha_filler_pipeline.py` | Complete analysis as a single module. Reproduces every table and figure in the paper. |
| `HA_Filler_analysis.ipynb` | The same analysis as an annotated notebook, run step by step with explanatory text. |
| `requirements.txt` | Library versions. |
| `results/tables/` | Aggregate results, including fold-level performance estimates for every classifier. |

## Running the analysis

```bash
pip install -r requirements.txt
python ha_filler_pipeline.py --data "HA filler (1).sav" --outdir results
```

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--data` | `HA_filler.sav` | Path to the SPSS file |
| `--outdir` | `results` | Output directory |
| `--imputations` | 5 | Number of imputed datasets |
| `--n-iter` | 20 | Hyperparameter search budget |
| `--seed` | 42 | Global random seed |
| `--fast` | off | Smoke test: 2 imputations, 3 classifiers, small search |

The notebook is the easier entry point. Set `DATA_PATH` in the configuration cell,
leave `FAST_MODE = True` for a first pass that takes a few minutes, then set it to
`False` for the full analysis.

## Method

Nested cross-validation: a stratified 5-fold outer loop and a stratified 3-fold
inner loop, with hyperparameters selected by ROC-AUC. No separate hold-out
partition is used, and no feature selection is performed at any stage.

The complete preprocessing sequence — ordinal encoding of the Allergan severity
scales, visit-wise interpolation and carry-forward, multivariate imputation by
chained equations, construction of temporal change features, and one-hot encoding
of nominal variables — is implemented as a single pipeline fitted **exclusively on
the training partition of each fold**. No information from a held-out fold
contributes to any preprocessing parameter.

Because multivariate imputation is stochastic, the whole nested procedure is
repeated across five independently imputed datasets with the outer fold assignment
held fixed, giving 25 performance estimates per classifier.

Six classifiers are compared: penalised (elastic-net) logistic regression, L2
logistic regression, Random Forest, AdaBoost, XGBoost, and an RBF-kernel support
vector machine.

## Reproducibility

A fixed random seed controls all cross-validation splits, imputation, bootstrap
resampling and SHAP sampling. The run records the exact library versions in
`run_metadata.json`. Re-running the pipeline on the same data with the same seed
reproduces every reported value.

## Data availability

The individual patient data are **not** included in this repository: the informed
consent obtained did not cover public release of individual records. Aggregate
results sufficient to verify every analysis — including the out-of-fold performance
estimate for each classifier in each cross-validation fold — are provided in
`results/tables/`.

## Licence

MIT. See `LICENSE`.
