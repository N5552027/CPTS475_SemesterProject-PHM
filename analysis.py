"""Reproducible, group-separated preliminary catalog classification."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
FEATURES = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff', 'koi_srad', 'koi_slogg', 'koi_kepmag']
LOG_COLUMNS = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_srad']

def prepare(raw):
    """Deterministic validity rules only; learned imputation stays in pipelines."""
    df = raw.drop_duplicates().copy()
    if not df['koi_disposition'].isin([0, 1, 2]).all():
        raise ValueError('Unexpected or missing target')
    if df[['kepid', 'kepoi_name']].isna().any().any():
        raise ValueError('Missing identifier')
    if df['kepoi_name'].duplicated().any():
        raise ValueError('Repeated candidate identifiers require investigation')
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    for col in ['koi_period', 'koi_prad', 'koi_teq', 'koi_steff', 'koi_srad']:
        df.loc[df[col] <= 0, col] = np.nan
    df.loc[df.koi_insol < 0, 'koi_insol'] = np.nan
    return df

def features(df, engineered=False):
    x = df[FEATURES].copy()
    if engineered:
        for col in LOG_COLUMNS:
            x[col] = np.log1p(x[col])
        # Catalog radii are Earth radii and solar radii respectively.
        # 6371/695700 converts Earth radii to solar radii (approximate).
        ratio = df.koi_prad * (6371.0 / 695700.0) / df.koi_srad
        x['log1p_radius_ratio'] = np.log1p(ratio)
        # Relative luminosity proxy from Stefan-Boltzmann scaling.
        x['log1p_luminosity_proxy'] = np.log1p(df.koi_srad ** 2 * (df.koi_steff / 5772.0) ** 4)
    return x

def run():
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT / 'data/exoplanets_data.csv')
    df = prepare(raw)
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.30, random_state=475).split(df, groups=df.kepid))
    assert not set(df.iloc[train].kepid) & set(df.iloc[test].kepid)
    assert set(df.iloc[train].koi_disposition) == set(df.iloc[test].koi_disposition) == {0, 1, 2}
    pd.DataFrame({'kepoi_name':df.kepoi_name, 'kepid':df.kepid,
                  'split':np.where(np.isin(np.arange(len(df)), test), 'evaluation', 'training')}).to_csv(out/'split.csv', index=False)
    audit = {'rows':len(raw), 'columns':len(raw.columns), 'duplicate_rows':int(raw.duplicated().sum()),
             'unique_stars':int(df.kepid.nunique()), 'raw_missing':raw.isna().sum().to_dict(),
             'prepared_missing':df[FEATURES].isna().sum().to_dict(),
             'class_counts':df.koi_disposition.value_counts().sort_index().to_dict(),
             'training_rows':len(train), 'evaluation_rows':len(test), 'star_overlap':0,
             'training_class_counts':df.iloc[train].koi_disposition.value_counts().sort_index().to_dict(),
             'evaluation_class_counts':df.iloc[test].koi_disposition.value_counts().sort_index().to_dict()}
    (out/'audit.json').write_text(json.dumps(audit, indent=2))
    df[FEATURES].describe(percentiles=[.01,.5,.99]).to_csv(out/'numeric_summary.csv')
    plt.rcParams.update({'figure.dpi':140, 'axes.spines.top':False, 'axes.spines.right':False})
    colors = ['#4477AA', '#EE6677', '#228833']
    fig, ax = plt.subplots(figsize=(7,4))
    counts = df.koi_disposition.value_counts().sort_index()
    bars = ax.bar([f'Class {i}' for i in counts.index], counts.values, color=colors)
    ax.bar_label(bars, labels=[f'{n:,} ({n/len(df):.1%})' for n in counts], padding=4)
    ax.set(title='Catalog target distribution (all records)', ylabel='Number of candidate records', ylim=(0,5600))
    fig.tight_layout(); fig.savefig(out/'class_distribution.png'); plt.close(fig)
    # Feature exploration restricted to training observations.
    fig, ax = plt.subplots(figsize=(8,5))
    for label, color in zip([0,1,2], colors):
        subset = df.iloc[train].query('koi_disposition == @label')
        ax.scatter(subset.koi_period, subset.koi_prad, s=9, alpha=.35, c=color, label=f'Class {label}')
    ax.set(xscale='log', yscale='log', xlabel='Orbital period (days)', ylabel='Planet radius (Earth radii)', title='Period and radius by class (training records)')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'period_radius.png'); plt.close(fig)
    estimators = {'Majority baseline':DummyClassifier(strategy='most_frequent'),
                  'Decision Tree':DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, class_weight='balanced', random_state=475),
                  'Random Forest':RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight='balanced', random_state=475, n_jobs=-1),
                  'Logistic Regression':LogisticRegression(max_iter=3000, class_weight='balanced', random_state=475)}
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=475)
    y = df.koi_disposition
    rows, reports = [], {}
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    plot_index = 0
    for engineered in [False, True]:
        x = features(df, engineered)
        for name, estimator in estimators.items():
            if engineered and name == 'Majority baseline':
                continue
            pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), clone(estimator))
            scores = cross_val_score(pipeline, x.iloc[train], y.iloc[train], groups=df.iloc[train].kepid,
                                     cv=cv, scoring='f1_macro', n_jobs=1)
            pipeline.fit(x.iloc[train], y.iloc[train])
            predicted = pipeline.predict(x.iloc[test])
            variant = 'engineered' if engineered else 'original'
            key = f'{name} / {variant}'
            rows.append({'model':name, 'features':variant, 'cv_macro_f1_mean':scores.mean(),
                         'cv_macro_f1_std':scores.std(), 'train_macro_f1':f1_score(y.iloc[train], pipeline.predict(x.iloc[train]), average='macro'),
                         'evaluation_accuracy':accuracy_score(y.iloc[test], predicted),
                         'evaluation_macro_f1':f1_score(y.iloc[test], predicted, average='macro')})
            reports[key] = classification_report(y.iloc[test], predicted, output_dict=True, zero_division=0)
            cm = confusion_matrix(y.iloc[test], predicted, labels=[0,1,2])
            reports[key]['confusion_matrix'] = cm.tolist()
            reports[key]['one_vs_rest_counts'] = {str(label):{'TP':int(cm[i,i]), 'FP':int(cm[:,i].sum()-cm[i,i]),
                'FN':int(cm[i,:].sum()-cm[i,i]), 'TN':int(cm.sum()-cm[:,i].sum()-cm[i,:].sum()+cm[i,i])} for i,label in enumerate([0,1,2])}
            pd.DataFrame({'kepoi_name':df.iloc[test].kepoi_name, 'actual':y.iloc[test], 'predicted':predicted}).to_csv(out/(name.lower().replace(' ','_')+'_'+variant+'_predictions.csv'),index=False)
            if engineered:
                ConfusionMatrixDisplay(cm, display_labels=[0,1,2]).plot(ax=axes[plot_index], colorbar=False, cmap='Blues')
                axes[plot_index].set_title(name); plot_index += 1
    fig.suptitle('Evaluation confusion matrices — engineered features')
    fig.tight_layout(); fig.savefig(out/'confusion_matrices.png'); plt.close(fig)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/'metrics.csv', index=False)
    (out/'classification_reports.json').write_text(json.dumps(reports, indent=2))
    print(json.dumps(audit, indent=2))
    print(metrics.round(4).to_string(index=False))
    return metrics

if __name__ == '__main__':
    run()
