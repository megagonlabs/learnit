import os
import unittest

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from learnit.autoconverter.autoconverter import AutoConverter
from learnit.autolearn.autolearn import AutoLearn
from learnit.autolearn.evaluate import Evaluate, ErrorSummary


class EvaluateTestCase(unittest.TestCase):
    def setUp(self):
        self.df1 = pd.read_csv('data/train.csv')
        self.assertTrue(True)
        ac1 = AutoConverter(target='Survived')
        self.assertTrue(True)
        X1, y1 = ac1.fit_transform(self.df1)
        al1 = AutoLearn(level=1)
        al1.learn(X1, y1)
        self.e1 = Evaluate(ac=ac1, alearn=al1)
        self.assertTrue(True)

        clf1 = LogisticRegression()
        clf1.fit(X1, y1)
        self.e1a = Evaluate(ac=ac1, alearn=clf1)

        with self.assertRaises(ValueError):
            Evaluate(alearn=al1)

        self.e1b = Evaluate(alearn=al1,
                            feature_names=ac1.feature_names)

        data = datasets.load_iris()
        self.df2 = pd.DataFrame(np.c_[data.target.reshape(-1, 1),
                                      data.data],
                                columns=["class"] + data.feature_names)
        ac2 = AutoConverter(target="class")
        al2 = AutoLearn(level=1)
        X2, y2 = ac2.fit_transform(self.df2)
        al2.learn(X2, y2)
        self.e2 = Evaluate(ac=ac2, alearn=al2)

        clf2 = LogisticRegression()
        clf2.fit(X2, y2)
        self.e2a = Evaluate(ac=ac2, alearn=clf2)

        # subtable
        dirpath = "data/kaggle-kkbox-churn-prediction-challenge-1k"
        members_df = pd.read_csv(os.path.join(dirpath, "members_train.csv"))
        transactions_df = pd.read_csv(
            os.path.join(dirpath, "transactions.csv"))
        user_logs_df = pd.read_csv(os.path.join(dirpath, "user_logs.csv"))

        subtables3 = {"transactions": {"table": transactions_df,
                                       "link_key": "msno",
                                       "group_key": "msno"},
                      "user_logs": {"table": user_logs_df,
                                    "link_key": "msno",
                                    "group_key": "msno"}}

        ac3 = AutoConverter(target="is_churn")
        X3, y3 = ac3.fit_transform(df=members_df,
                                   subtables=subtables3)
        al3 = AutoLearn(level=1)
        al3.learn(X3, y3)
        self.e3 = Evaluate(ac=ac3, alearn=al3)

        self.df4 = members_df
        ac4 = AutoConverter(target="is_churn", task_type="regression")
        X4, y4 = ac4.fit_transform(df=members_df)
        al4 = AutoLearn(level=1, task="regression")
        al4.learn(X4, y4)
        e4 = Evaluate(alearn=al4, ac=ac4)
        self.e4 = e4

    def test_calculate_column_importance(self):
        for e in [self.e1,
                  self.e1a,
                  self.e2,
                  self.e2a,
                  self.e3,
                  self.e4]:
            try:
                e.calculate_column_importance()
            except Exception as e:
                self.fail(str(e))

    def test_evaluate(self):
        for e in [self.e1,
                  self.e1a,
                  self.e2,
                  self.e2a,
                  self.e3,
                  self.e4]:
            orig_eval_s = e.evaluate_performance()
            col_imp_df = e.calculate_column_importance()
            self.assertEqual(orig_eval_s.index.tolist(),
                             col_imp_df.columns.tolist())

        # They should raise Errors as X and y are not given
        with self.assertRaises(ValueError):
            self.e1b.evaluate_performance()

        with self.assertRaises(ValueError):
            self.e1b.calculate_column_importance()

    def test_get_top_column(self):
        self.assertEqual(5, len(self.e1.get_top_columns(n=5)))
        for table_colname in self.e3.get_top_columns(n=3):
            tablename = table_colname.split("..")[0]
            self.assertTrue(tablename in
                            list(self.e3.ac.subtables_.keys()) + ["main"])

    def test_get_mispredictions(self):
        for e, df in [(self.e1, self.df1),
                      (self.e1a, self.df1),
                      (self.e2, self.df2),
                      (self.e2a, self.df2)]:
            mispred_df = e.get_mispredictions(df)
            orig_colset = set(df.columns.tolist())
            mispred_colset = set(mispred_df.columns.tolist())

            # All columns in mispred_df should be in df
            self.assertEqual(len(mispred_colset & orig_colset),
                             len(mispred_colset))

    def test_stratify_errors(self):
        for e, df in [(self.e1, self.df1),
                      (self.e1a, self.df1)]:
            es = e.stratify_errors(df)
            self.assertIsNotNone(es)
            self.assertIsInstance(es, ErrorSummary)
            self.assertIsNotNone(es.diversity)
            self.assertIsNotNone(es.error_dist)
            self.assertIsNotNone(es.errors)
            self.assertEqual(es.error_dist.index.levels[0].tolist(),
                             es.diversity.index.tolist())

        # None should be returned for the Iris dataset
        self.assertIsNone(self.e2.stratify_errors(self.df2))
        self.assertIsNone(self.e2a.stratify_errors(self.df2))

    def test_get_explanations(self):
        e_df1 = self.e1.get_explanations(self.df1)
        self.assertEqual(e_df1.shape[0], self.df1.shape[0])
        e_df1a = self.e1a.get_explanations(self.df1)
        self.assertEqual(e_df1a.shape[0], self.df1.shape[0])
        e_df2 = self.e2.get_explanations(self.df2)
        self.assertEqual(e_df2.shape[0], self.df2.shape[0])
        e_df2a = self.e2a.get_explanations(self.df2)
        self.assertEqual(e_df2a.shape[0], self.df2.shape[0])
