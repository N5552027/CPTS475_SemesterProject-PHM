import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from analysis import ROOT, FEATURES, prepare, features

class AnalysisChecks(unittest.TestCase):
    def setUp(self):
        self.raw = pd.read_csv(ROOT / 'data/exoplanets_data.csv')

    def test_invalid_measurements_become_missing(self):
        sample = self.raw.head(4).copy()
        sample.loc[0, 'koi_srad'] = 0
        sample.loc[1, 'koi_period'] = -1
        sample.loc[2, 'koi_prad'] = np.inf
        clean = prepare(sample)
        self.assertTrue(clean.loc[0:2, FEATURES].isna().any(axis=1).all())
        self.assertFalse(np.isinf(features(clean, True).to_numpy()).any())

    def test_identifiers_score_and_target_excluded(self):
        x = features(prepare(self.raw), True)
        self.assertFalse(set(x) & {'kepid', 'kepoi_name', 'koi_disposition', 'koi_score'})
        self.assertTrue(np.isfinite(x.to_numpy()).all())

    def test_group_separation_and_saved_predictions(self):
        df = prepare(self.raw)
        train, test = next(GroupShuffleSplit(test_size=.3, random_state=475).split(df, groups=df.kepid))
        self.assertFalse(set(df.iloc[train].kepid) & set(df.iloc[test].kepid))
        for path in (ROOT/'results').glob('*_predictions.csv'):
            prediction = pd.read_csv(path)
            self.assertEqual(set(prediction.kepoi_name), set(df.iloc[test].kepoi_name))
            self.assertEqual(len(prediction), len(test))
            self.assertTrue(prediction.predicted.isin([0,1,2]).all())

    def test_radius_conversion(self):
        df = prepare(self.raw.head(1))
        df.loc[0, 'koi_prad'] = 695700 / 6371
        df.loc[0, 'koi_srad'] = 1
        self.assertAlmostEqual(features(df, True).loc[0, 'log1p_radius_ratio'], np.log(2))

if __name__ == '__main__':
    unittest.main()
