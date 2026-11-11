# Detecting Exoplanets from Stellar Light Curves

Team Project Hail Mary: Finley Blaylock and Noah Manuel.

This milestone classifies the three encoded Kepler catalog dispositions using derived planetary and stellar measurements. The supplied CSV is **not raw light-curve data**. Results describe catalog-label prediction, not confirmed new planet discoveries. The meanings of numeric labels 0/1/2 remain unverified; do not assign astrophysical names based only on class frequencies.

## Run

Tested with Python 3.14.3 and the versions in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python analysis.py
python -m unittest -v test_analysis.py
```

Open `ProjectWorkbook.ipynb` in a notebook-capable editor and run all cells from the repository directory. It preserves the partner cleaning workflow and trains the three models in Parts 3?5. Its cleaned-data results are saved in `results_cleaned/`; `python analysis.py` reproduces the earlier full-data experiment in `results/`. These experiments use different samples. No downloads are required to run the analysis. `results/` contains the reproducible split, audit, seven model/feature comparisons, per-class reports, predictions, and three figures. `MILESTONE_UPDATE.md` is a draft to review before posting; team contributions have been filled in from the team?s account. No post has been submitted.

## Method and limitations

- Validate types, finite measurements, positive period/radii/temperatures, and nonnegative insolation. No original observations required removal or imputation. Exact duplicate removal is supported; no duplicates were found. Raw CSV remains unchanged.
- Retain large positive outliers pending provenance review. Summary quantiles are exported. Extreme fitted radii may describe false-positive signals, so arbitrary clipping could erase useful information. Log transforms reduce scale differences but do not establish physical plausibility.
- Exclude `kepid`, `kepoi_name`, and disposition-confidence `koi_score` from predictors. Keep star IDs only for grouping. Median imputation and scaling are fitted independently in each training fold.
- Fixed seed 475; approximately 70/30 split by **star**, yielding 6,681 training and 2,883 evaluation rows. Three-fold stratified group cross-validation uses training records only. Group separation is prioritized over an exact row percentage.
- Eight original features versus ten engineered features: log1p transforms of period, planet radius, equilibrium temperature, insolation, and stellar radius; log1p radius ratio and relative luminosity proxy. These are derived from existing catalog measurements, not independent observations. No feature selection or hyperparameter search used the evaluation labels.
- Fixed preliminary hyperparameters; balanced class weights for the three trained classifiers. Original Logistic Regression inputs are also standardized. Report macro-F1 to give equal weight to all three classes.
- Evaluation results are preliminary and have now been inspected. Further selection should use training cross-validation; reserve fresh evaluation data for a final unbiased estimate if repeated iteration is substantial. Three CV folds are not confidence intervals.

## Sources and outstanding provenance

- Dataset source recorded in the original notebook: [Kaggle Kepler Exoplanet Dataset](https://www.kaggle.com/datasets/gauravkumar2525/kepler-exoplanet-dataset/data).
- [NASA archive column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html) describe disposition confidence and catalog measurement units. The dataset publisher's numeric encoding and earlier cleaning steps are not supplied in this repository. Verify both before interpreting named classes; current NASA dispositions may have changed since this CSV was produced.

Peer feedback must be based on a real classmate's update: identify one specific strength, ask a concrete data/method question, and offer one actionable recommendation. Do not submit generic feedback without reviewing their analysis.
