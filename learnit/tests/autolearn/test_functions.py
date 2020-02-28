import math
import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression

from learnit.autoconverter.autoconverter import AutoConverter
from learnit.autolearn.functions import (__run_cross_validation as run_cv,
                                         run_validation)
from learnit.autolearn.blueprints import GBClassifierGridSearchCV


class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('data/train.csv')
        ac = AutoConverter(target='Survived')
        X, y = ac.fit_transform(self.df)
        self.X = X
        self.y = y

    def test_run_cross_validation(self):
        clf = LogisticRegression()
        cv_num = 6
        result_info = run_cv(
            self.X, self.y, clf=clf, metric='roc_auc', cv_num=cv_num)

        self.assertIsInstance(result_info, dict)
        self.assertIsInstance(result_info['cv_df'], pd.DataFrame)
        self.assertEqual(result_info['cv_df'].shape,
                         (cv_num, 2), 'wrond cv_df shape')
        self.assertEqual(result_info['y_error'].shape[0],
                         self.X.shape[0])
        self.assertEqual(math.floor(
            self.X.shape[0] / result_info['y_pred'].shape[0]), cv_num)
        clf2 = GBClassifierGridSearchCV
        result_info2 = run_validation(self.X, self.y, est=clf2, cv_num=1,
                                      n_jobs=2, metric='roc_auc')
        est = result_info2["sample_clf"].estimator
        if "XGB" in est.__class__.__name__:
            self.assertEqual(est.n_jobs, 2)
        else:
            self.assertEqual(result_info2["sample_clf"].n_jobs, 2)

    def tearDown(self):
        pass
