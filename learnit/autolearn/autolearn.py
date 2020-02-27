# -*- coding: utf-8 -*-
import os
import sys
from time import time

import yaml

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from tabulate import tabulate

from learnit.autolearn.blueprints import ClassifierCatalog
from learnit.autolearn.blueprints import MultiClassifierCatalog
from learnit.autolearn.blueprints import RegressorCatalog

from autosklearn.regression import AutoSklearnRegressor
from autosklearn.classification import AutoSklearnClassifier
# TODO(Bublin) Load Metric Function from autosklearn.metrics~
from autosklearn.metrics import roc_auc, log_loss

if sys.version_info.major == 3:
    import pickle
else:
    import cPickle as pickle


class AutoLearnBase(object):
    '''Base class for AutoConverter and AutoRegressor.'''

    def __init__(self):
        """Init function."""

    def save(self,
             filepath,
             overwrite=False):
        """Save AutoLearn object as pickle file

        Args:
            filepath (str): Output pickle filepath
            overwrite (bool): Overwrites a file with the same name if true

        Returns:
            success_flag (bool)

        """
        if not overwrite and os.path.exists(filepath):
            # TODO(Yoshi): Warning handling
            print("File already exists. Skip.")
            return False

        with open(filepath, "wb") as fout:
            pickle.dump(self, fout)

        return True

    @classmethod
    def load(cls,
             filepath):
        """Load AutoLearn object from pickle file

        Args:
            filepath (str): Input pickle filepath

        Returns:
            AutoLearn object

        """
        with open(filepath, "rb") as fin:
            obj = pickle.load(fin)
        if not isinstance(obj, cls):
            raise TypeError('Loaded object %s is not an instance of %s '
                            'class' % (type(obj), cls))

        return obj

    def display(self, tab=True):
        raise NotImplementedError

    def fit(self, X, y):
        """fit

        Args:
            X (np.array): feature matrix
            y (np.array): label vector

        """
        raise NotImplementedError

    def predict(self, X):
        """Predict by self.best_clf

        Args:
            X:
            y:

        Returns:
            pred:

        """
        raise NotImplementedError

    def predict_proba(self, X):
        """Predict probabilities by self.best_clf

        Args:
            X:
            y:

        Returns:
            pred:

        """
        raise NotImplementedError


class AutoClassifier(AutoLearnBase):
    def __init__(self,
                 level=1,
                 task_type='auto',
                 metric='auto',
                 cv_num=5,
                 validation_ratio=0.2,
                 pos_label=1,
                 n_jobs=1,
                 time_left_for_this_task=600,
                 per_run_time_limit=20,
                 suppress_warning=True):
        """init function

        Args:
            level (int): 1-5
            task_type (string): ['auto', 'binary', 'multi']
            metric (string): evaluation metric
                             ['auto', 'roc_auc', 'neg_log_loss']
            cv_num: cross valudation number
            pos_label (int): positive label
            n_jobs: number of jobs for parallelization
            verbose: verbose level
            customized_clf_list (list): default: None

        """
        assert task_type in ['auto', 'binary', 'multi']
        assert metric in ['auto', 'roc_auc', 'log_loss']

        self.clf = None
        self.level = level
        self.task_type = task_type
        self.metric = metric
        self.cv_num = cv_num
        self.validation_ratio = validation_ratio
        self.pos_label = pos_label
        self.n_jobs = n_jobs
        self.trained = False
        self.info = None
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.suppress_warning = suppress_warning

    # TODO(Bublin) These Check can be done by auto-sklearn
    def type_task(self, y):
        """Determine prediction task for classification.

        Args:
            y (np.array): label vector

        Returns:
            task_type: either 'binary' or 'multi'
        """
        y_unique = np.unique(y)

        # Automatically determine the task type
        if self.task_type == 'auto':
            if len(y_unique) == 2:
                self.task_type = 'binary'
            else:
                if y_unique.all() == np.arange(len(y_unique)).all():
                    self.task_type = 'multi'
                else:
                    self.task_type = 'multi'
                    print('Warning! number of values might suggest the use'
                          ' of regression instead of classification')
            if 'float' in str(y.dtype):
                # we should raise an exception here
                print('Warning! float values might suggest the use'
                      ' of regression instead of classification')

        if self.task_type not in ['binary', 'multi']:
            raise ValueError('Invalid task type: %s' % self.task_type)

    def pre_learn(self, X, y):
        """Take care of configuration before fitting models.

        Args:
            X (np.array): feature matrix
            y (np.array): label vector

        """
        # Check prediction type
        self.type_task(y)

        # Check metric
        if self.metric == 'auto':
            if self.task_type == 'binary':
                self.metric = 'roc_auc'
            elif self.task_type == 'multi':
                self.metric = 'log_loss'

        assert self.task_type in ['binary', 'multi']
        # Currently, only support auc_roc for binary, neg_log_loss for multi
        if self.task_type == 'binary':
            assert self.metric in ['roc_auc']
        else:
            assert self.metric in ['log_loss']

        assert self.cv_num >= 1
        clf_config = {
            # No Preprocessing
            "include_preprocessors": ["no_preprocessing"],
            "n_jobs": self.n_jobs,
            "time_left_for_this_task": self.time_left_for_this_task,
            "per_run_time_limit": self.per_run_time_limit
        }
        if self.cv_num == 1:
            clf_config["resampling_strategy"] = "holdout"
            clf_config["resampling_strategy_arguments"] = {'train_size': 1-self.validation_ratio}
        else:
            clf_config["resampling_strategy"] = "cv"
            clf_config["resampling_strategy_arguments"] = {'folds': self.cv_num}
        name = "level{}".format(self.level)
        if self.task_type == "multi":
            catalog_dict = vars(MultiClassifierCatalog)
        else:
            catalog_dict = vars(ClassifierCatalog)
        if name not in catalog_dict:
            raise ValueError('level %s not supported' % self.level)
        clf_config["include_estimators"] = catalog_dict[name]
        if self.suppress_warning:
            with open(os.path.join(os.path.dirname(__file__), 'suppress_logging.yaml'),
                      'r') as fh:
                logging_config = yaml.safe_load(fh)
            clf_config["logging_config"] = logging_config

        self.clf = AutoSklearnClassifier(**clf_config)

    def fit(self, X, y):
        """fit

        Args:
            X (np.array): feature matrix
            y (np.array): label vector

        Returns:
            {"clf_name": clf_name,
             "clf": clf,
             "eval": eval_df}

        # [Discussion] Do we want to make it a class object?
        #       or just dict is fine enough.

        """
        t1 = time()
        self.info = {}
        self.info["dataset_size"], self.info["feature_size"] = X.shape

        # Pre-training configuration
        self.pre_learn(X, y)
        try:
            metric_module = globals()[self.metric]
            self.clf.fit(X.copy(), y.copy(), metric=metric_module)
            self.info["name"] = self.clf.show_models()
            self.info["clf"] = self.clf
            if self.cv_num > 1:
                self.clf.refit(X.copy(), y.copy())

        except Exception as e:
            # TODO(Yoshi): Handling exception
            print(e)
            return None

        self.trained = True
        t2 = time()

        self.info["training_time"] = t2 - t1

        return self.info

    def predict(self, X):
        """Predict by self.best_clf

        Args:
            X:
            y:

        Returns:
            pred:

        """
        assert self.trained
        assert hasattr(self.clf, 'predict')
        return self.clf.predict(X)

    def predict_proba(self, X):
        """Predict probabilities by self.best_clf

        Args:
            X:
            y:

        Returns:
            pred:

        """
        assert self.trained
        assert hasattr(self.clf, 'predict_proba')
        return self.clf.predict_proba(X)

    def display(self, tab=True):

        """Fancy display function for al.info

        Args:
            tab (boolean): flag to use tabulate. Default: True

        Returns:
            info
            clf_metrics (pd.DataFrame): the metrics dataframe

        """

        # TODO(Yoshi): If print this message for self.trained == False,
        # "Model is trained" info below is redundant (as it is always True)
        if not self.trained:
            print('Model is not trained')
            return

        metric_names = {'Model is trained': self.trained,
                        'Best classifier': self.clf.show_models(),
                        'Evaluation metric': self.metric,
                        'Dataset size': self.info["dataset_size"],
                        '# of features': self.info["feature_size"],
                        'Classifier set level': self.level,
                        'Including classifiers': list(map(lambda x: x[1], self.clf.get_models_with_weights())),
                        'Training time': "{:.2f} sec.".format(self.info["training_time"])}

        features_df = pd.DataFrame(list(metric_names.items()),
                                   columns=['metric',
                                            'value']).set_index('metric')
        if tab:
            print(tabulate(features_df, headers='keys', tablefmt='psql'))
        else:
            features_df.head()

        return features_df


class AutoLearn(object):
    '''Wrapper class for AutoRegressor and AutoClassifier.'''
    def __init__(self,
                 task='classification',
                 level=1,
                 metric='auto',
                 cv_num=5,
                 validation_ratio=0.2,
                 pos_label=1,
                 n_jobs=1,
                 time_left_for_this_task=600,
                 per_run_time_limit=20,
                 suppress_warning=True
                 ):
        if task == 'regression':
            self.task = 'regression'
            self.learner = AutoRegressor(
                level=level,
                metric='mean_absolute_error',
                cv_num=cv_num,
                validation_ratio=validation_ratio,
                n_jobs=n_jobs,
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                suppress_warning=suppress_warning
            )

        elif task == 'classification':
            self.task = 'classification'
            self.learner = AutoClassifier(
                level=level,
                metric=metric,
                cv_num=cv_num,
                task_type='auto',
                validation_ratio=validation_ratio,
                pos_label=pos_label,
                n_jobs=n_jobs,
                time_left_for_this_task=time_left_for_this_task,
                per_run_time_limit=per_run_time_limit,
                suppress_warning=suppress_warning
            )

        else:
            raise ValueError('Wrong task_type = %s' % task)

    def fit(self, X, y):
        self.info = self.learner.fit(X, y)
        self.trained = self.learner.trained
        return self.info

    def learn(self, X, y):
        '''Left as an alias for fit, does the same.'''
        return self.fit(X, y)

    def predict(self, X):
        return self.learner.predict(X)

    def predict_proba(self, X):
        # TODO(Kate): what happens with predict_proba in case of regression?
        # NotImplementedError?
        return self.learner.predict_proba(X)

    def display(self):
        return self.learner.display()

    def save(self, filepath, overwrite=False):
        return self.learner.save(filepath, overwrite)

    @classmethod
    def load(self, filepath):
        return AutoLearnBase.load(filepath)

    def pre_learn(self):
        return self.learner.pre_learn()


class AutoRegressor(AutoLearnBase):
    def __init__(self, level, metric='mean_absolute_error', cv_num=5,
                 validation_ratio=0.2, n_jobs=1,
                 time_left_for_this_task=600,
                 per_run_time_limit=20, suppress_warning=True):
        """Init

        Args:
            level (int): 1:2
            metric (string): evaluation metric used for cross validation
            cv_num (int): cross validation number
            verbose: verbose level

        Returns:

        """
        self.clf = None
        self.level = level
        self.metric = metric
        self.cv_num = cv_num
        self.validation_ratio = validation_ratio
        self.n_jobs = n_jobs
        self.trained = False
        self.info = None
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.suppress_warning = suppress_warning

    def fit(self, X, y):
        """Fitting the regressor

        Args:
            X (np.array): feature matrix
            y (np.array): label vector

        Returns:
            {"reg_name": reg_name,
             "reg": reg,
             "eval": eval_df}
        WHAT IS EVAL DF?
        """

        catalog_dict = vars(RegressorCatalog)
        name = "level{}".format(self.level)
        if name not in catalog_dict:
            raise ValueError('level %s not supported' % self.level)
        clf_config = {
            # No Preprocessing
            "include_preprocessors": ["no_preprocessing"],
            "n_jobs": self.n_jobs,
            "time_left_for_this_task": self.time_left_for_this_task,
            "per_run_time_limit": self.per_run_time_limit
        }
        if self.cv_num == 1:
            clf_config["resampling_strategy"] = "holdout"
            clf_config["resampling_strategy_arguments"] = {
                'train_size': 1 - self.validation_ratio}
        else:
            clf_config["resampling_strategy"] = "cv"
            clf_config["resampling_strategy_arguments"] = {
                'folds': self.cv_num}
        clf_config["include_estimators"] = catalog_dict[name]
        if self.suppress_warning:
            with open(os.path.join(os.path.dirname(__file__), 'suppress_logging.yaml'),
                      'r') as fh:
                logging_config = yaml.safe_load(fh)
            clf_config["logging_config"] = logging_config
        self.clf = AutoSklearnRegressor(**clf_config)
        self.info = {}
        self.info["dataset_size"], self.info["feature_size"] = X.shape
        t1 = time()
        try:
            # autosklearn can not use any metrics in Regressor
            self.clf.fit(X.copy(), y.copy())
            self.info["name"] = "autosklearn"
            self.info["clf"] = self.clf
            if self.cv_num > 1:
                self.clf.refit(X.copy(), y.copy())

        except Exception as e:
            # TODO(Yoshi): Handling exception
            print(e)
            return None

        self.trained = True
        t2 = time()

        self.info["training_time"] = t2 - t1

        return self.info

    def predict(self, X):
        """Predict by self.best_est

        Args:
            X:
            y:

        Returns:
            pred:

        """
        if not self.trained:
            raise NotFittedError('Regressor not fitted')
        assert hasattr(self.clf, 'predict')
        return self.clf.predict(X)

    def display(self, tab=True):
        """Fancy display function for al.info

        Args:
            tab (boolean): flag to use tabulate. Default: True

        Returns:
            info
            est_metrics (pd.DataFrame): the metrics dataframe

        """

        if not self.trained:
            print('Model is not trained')
            return

        metric_names = {'Model is trained': self.trained,
                        'Best classifier': self.clf.show_models(),
                        'Evaluation metric': self.metric,
                        'Dataset size': self.info["dataset_size"],
                        '# of features': self.info["feature_size"],
                        'Classifier set level': self.level,
                        'Including classifiers': list(map(lambda x: x[1],
                                                          self.clf.get_models_with_weights())),
                        'Training time': "{:.2f} sec.".format(
                            self.info["training_time"])}

        features_df = pd.DataFrame(list(metric_names.items()),
                                   columns=['metric',
                                            'value']).set_index('metric')
        if tab:
            print(tabulate(features_df, headers='keys', tablefmt='psql'))
        else:
            features_df.head()

        return features_df
