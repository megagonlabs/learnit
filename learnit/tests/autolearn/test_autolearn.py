import os
import shutil
import unittest

import numpy as np
from sklearn import datasets

from learnit.autolearn.autolearn import AutoLearn

TIME_CONFIG = {
    "time_left_for_this_task": 30,
    "per_run_time_limit": 5
}


class AutoLearnTestCase(unittest.TestCase):
    def setUp(self):
        self.breast_cancer = datasets.load_breast_cancer()
        self.iris_data = datasets.load_iris()
        self.boston_data = datasets.load_boston()

    def _test_pre_learn(self):
        al1 = AutoLearn(**TIME_CONFIG)
        al1.pre_learn(self.breast_cancer.data,
                      self.breast_cancer.target)
        self.assertEqual(al1.task_type, 'binary')
        self.assertEqual(al1.metric, 'roc_auc')

        al2 = AutoLearn(**TIME_CONFIG)
        al2.pre_learn(self.iris_data.data,
                      self.iris_data.target)
        self.assertEqual(al2.task_type, 'multi')
        self.assertEqual(al2.metric, 'log_loss')

        al3 = AutoLearn(**TIME_CONFIG)
        with self.assertRaises(AssertionError):
            al3.pre_learn(self.boston_data.data,
                          self.boston_data.target)

    def test_learn(self):
        al1 = AutoLearn(level=1, **TIME_CONFIG)
        al1.learn(self.breast_cancer.data,
                  self.breast_cancer.target)
        al2 = AutoLearn(level=1, **TIME_CONFIG)
        al2.learn(self.iris_data.data,
                  self.iris_data.target)
        al3 = AutoLearn(level=1, cv_num=1, validation_ratio=0.2,
                        **TIME_CONFIG)
        al3.learn(self.breast_cancer.data,
                  self.breast_cancer.target)
        al4 = AutoLearn(level=1, cv_num=1, validation_ratio=0.2,
                        **TIME_CONFIG)
        al4.learn(self.iris_data.data,
                  self.iris_data.target)
        al5 = AutoLearn(level=2, n_jobs=2, **TIME_CONFIG)
        al5.learn(self.breast_cancer.data,
                  self.breast_cancer.target)
        al6 = AutoLearn(level=2, n_jobs=2, **TIME_CONFIG)
        al6.learn(self.iris_data.data,
                  self.iris_data.target)

    def test_predict_functions(self):
        al1 = AutoLearn(level=1, **TIME_CONFIG)
        al1.learn(self.breast_cancer.data,
                  self.breast_cancer.target)
        self.assertEqual(al1.predict_proba(self.breast_cancer.data).shape,
                         (569, 2))

        al2 = AutoLearn(level=1, **TIME_CONFIG)
        al2.learn(self.iris_data.data,
                  self.iris_data.target)
        self.assertEqual(al2.predict_proba(self.iris_data.data).shape,
                         (150, 3))

    def test_regressor(self):
        al1 = AutoLearn(level=1, task='regression')
        al1.learn(self.boston_data.data, self.boston_data.target)
        self.assertEqual(al1.trained, True)

        al2 = AutoLearn(level=2, task='regression', cv_num=1)
        al2.learn(self.boston_data.data, self.boston_data.target)
        self.assertEqual(al2.trained, True)

    def test_save_load(self):
        """Call save() and load() functions."""
        al = AutoLearn(level=1, **TIME_CONFIG)
        al.learn(self.breast_cancer.data,
                 self.breast_cancer.target)
        tempdir = "__tmp__test__"
        filename = "al_test.pickle"
        filepath = os.path.join(tempdir, filename)
        if os.path.exists(tempdir):
            shutil.rmtree(tempdir)
        os.makedirs(tempdir)
        self.assertTrue(al.save(filepath))
        self.assertTrue(os.path.exists(filepath))
        self.assertFalse(al.save(filepath,
                                 overwrite=False))
        loaded_al = AutoLearn.load(filepath)
        self.assertEqual(al.info["name"], loaded_al.info["name"])
        shutil.rmtree(tempdir)
