# -*- coding: utf-8 -*-
import os
import sys
from time import time

import numpy as np
import pandas as pd
from progressbar import ProgressBar
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from tabulate import tabulate

from learnit.autolearn.blueprints import ClassifierCatalog
from learnit.autolearn.blueprints import MultiClassifierCatalog
from learnit.autolearn.blueprints import RegressorCatalog
from learnit.autolearn.functions import run_validation

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
                 verbose=0,
                 customized_clf_list=None):
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
        assert metric in ['auto', 'roc_auc', 'neg_log_loss']

        self.level = level
        self.task_type = task_type
        self.metric = metric
        self.cv_num = cv_num
        self.validation_ratio = validation_ratio
        self.pos_label = pos_label
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.clf_list = customized_clf_list
        self.trained = False
        self.info = None

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
                self.metric = 'neg_log_loss'

        assert self.task_type in ['binary', 'multi']
        # Currently, only support auc_roc for binary, neg_log_loss for multi
        if self.task_type == 'binary':
            assert self.metric in ['roc_auc']
        else:
            assert self.metric in ['neg_log_loss']

        # Load classifier blueprints
        if self.clf_list is None:
            if self.task_type == 'binary':
                catalog_dict = vars(ClassifierCatalog)
            elif self.task_type == 'multi':
                catalog_dict = vars(MultiClassifierCatalog)
            else:
                # TODO(Yoshi): Regression
                raise AssertionError(
                    "task_type not supported: {}".format(self.task_type))

            name = "level{}".format(self.level)
            assert name in catalog_dict
            self.clf_list = catalog_dict[name]

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
            print("Learning from data...")
            sys.stdout.flush()
            results_dict = {}
            clf_dict = {}
            bar = ProgressBar(widget_kwargs=dict(marker=u'█'))
            for name, clf in bar(self.clf_list):
                results = run_validation(
                    X, y, clf,
                    metric=self.metric,
                    cv_num=self.cv_num,
                    validation_ratio=self.validation_ratio,
                    pos_label=self.pos_label,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                )
                results_dict[name] = results
                # Note: clf should implement GridSearch so does not have to
                # conduct parameter search externally.
                final_clf = clone(clf)
                final_clf.fit(X, y)
                assert name not in clf_dict  # Ensure the uniqueness
                clf_dict[name] = final_clf

            max_name = None
            max_cv_df = None
            max_train_eval_df = None
            max_test_eval_df = None
            max_value = None

            for name, results in results_dict.items():
                cv_df = results["cv_df"]
                train_eval_df = results["train_eval_df"]
                test_eval_df = results["test_eval_df"]

                # TODO(Yoshi): User can choose evaluation metric for selection
                cur_value = cv_df['metric_test'].mean()
                if max_value is None or cur_value > max_value:
                    max_name = name
                    max_cv_df = cv_df
                    max_train_eval_df = train_eval_df
                    max_test_eval_df = test_eval_df
                    max_value = cur_value

            assert max_name in clf_dict
            self.best_clf = clf_dict[max_name]
            self.info["name"] = max_name
            self.info["clf"] = self.best_clf
            self.info["eval_df"] = max_cv_df
            self.info["train_eval_df"] = max_train_eval_df
            self.info["test_eval_df"] = max_test_eval_df

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
        assert hasattr(self.best_clf, 'predict')
        return self.best_clf.predict(X)

    def predict_proba(self, X):
        """Predict probabilities by self.best_clf

        Args:
            X:
            y:

        Returns:
            pred:

        """
        assert self.trained
        assert hasattr(self.best_clf, 'predict_proba')
        return self.best_clf.predict_proba(X)

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

        eval_metric = self.info['clf'].scoring
        clf_name = self.info['clf'].estimator.__class__.__name__
        metric_names = {'Model is trained': self.trained,
                        'Best classifier': clf_name,
                        'Evaluation metric': eval_metric,
                        'Dataset size': self.info["dataset_size"],
                        '# of features': self.info["feature_size"],
                        'Classifier set level': self.level,
                        'Including classifiers': list(map(lambda x: x[0],
                                                          self.clf_list)),
                        'Training time': "{:.2f} sec.".format(
                            self.info["training_time"])}

        features_df = pd.DataFrame(list(metric_names.items()),
                                   columns=['metric',
                                            'value']).set_index('metric')

        clf_metrics = ['Accuracy',
                       'Precision',
                       'Recall',
                       eval_metric]
        metrics_df = pd.DataFrame({'metric': clf_metrics}).set_index('metric')

        eval_test = self.info['eval_df']['metric_test'].mean()
        eval_train = self.info['eval_df']['metric_train'].mean()

        acc_test = self.info["test_eval_df"]["accuracy"].mean()
        acc_train = self.info["train_eval_df"]["accuracy"].mean()
        prec_test = self.info["test_eval_df"]["precision"].mean()
        prec_train = self.info["train_eval_df"]["precision"].mean()
        rec_test = self.info["test_eval_df"]["recall"].mean()
        rec_train = self.info["train_eval_df"]["recall"].mean()

        metrics_df['training set'] = [acc_train,
                                      prec_train,
                                      rec_train,
                                      eval_train]
        metrics_df['test set'] = [acc_test,
                                  prec_test,
                                  rec_test,
                                  eval_test]
        if tab:
            print(tabulate(features_df, headers='keys', tablefmt='psql'))
            print(tabulate(metrics_df, headers='keys', tablefmt='psql'))
        else:
            features_df.head()
            metrics_df.head()

        return features_df, metrics_df


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
                 verbose=0,
                 customized_clf_list=None):
        if task == 'regression':
            self.task = 'regression'
            self.learner = AutoRegressor(
                level=level,
                metric='neg_mean_absolute_error',
                cv_num=cv_num,
                validation_ratio=validation_ratio,
                n_jobs=n_jobs,
                verbose=verbose)

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
                verbose=verbose,
                customized_clf_list=customized_clf_list)

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
    def __init__(self, level, metric='neg_mean_absolute_error', cv_num=5,
                 verbose=0, validation_ratio=0.2, n_jobs=1):
        """Init

        Args:
            level (int): 1:2
            metric (string): evaluation metric used for cross validation
            cv_num (int): cross validation number
            verbose: verbose level

        Returns:

        """
        self.level = level
        self.metric = metric
        self.cv_num = cv_num
        self.verbose = verbose
        self.validation_ratio = validation_ratio
        self.n_jobs = n_jobs
        # TODO(Kate): decide if we want that and how we match
        # clf list in classification maybe
        # self.est_list = customized_reg_list
        self.trained = False
        self.info = None

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
        self.est_list = catalog_dict[name]

        t1 = time()
        self.info = {}
        self.info["dataset_size"], self.info["feature_size"] = X.shape

        print("Learning from data...")
        sys.stdout.flush()
        results_dict = {}
        est_dict = {}
        bar = ProgressBar(widget_kwargs=dict(marker=u'█'))
        for name, est in bar(self.est_list):
            results = run_validation(
                X, y, est,
                metric=self.metric,
                cv_num=self.cv_num,
                validation_ratio=self.validation_ratio,
                verbose=self.verbose)
            results_dict[name] = results
            final_est = clone(est)
            final_est.fit(X, y)
            assert name not in est_dict  # Ensure the uniqueness
            est_dict[name] = final_est

        max_name = None
        max_cv_df = None
        max_train_eval_df = None
        max_test_eval_df = None
        max_value = None

        for name, results in results_dict.items():
            cv_df = results["cv_df"]
            train_eval_df = results["train_eval_df"]
            test_eval_df = results["test_eval_df"]

            # TODO(Yoshi): User can choose evaluation metric for selection
            cur_value = cv_df['metric_test'].mean()
            if max_value is None or cur_value > max_value:
                max_name = name
                max_cv_df = cv_df
                max_train_eval_df = train_eval_df
                max_test_eval_df = test_eval_df
                max_value = cur_value

        assert max_name in est_dict
        self.best_est = est_dict[max_name]
        self.info["name"] = max_name
        self.info["clf"] = self.best_est
        self.info["eval_df"] = max_cv_df
        self.info["train_eval_df"] = max_train_eval_df
        self.info["test_eval_df"] = max_test_eval_df

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
        assert hasattr(self.best_est, 'predict')
        return self.best_est.predict(X)

    def display(self, tab=True):
        """Fancy display function for al.info

        Args:
            tab (boolean): flag to use tabulate. Default: True

        Returns:
            info
            est_metrics (pd.DataFrame): the metrics dataframe

        """

        # TODO(Yoshi): If print this message for self.trained == False,
        # "Model is trained" info below is redundant (as it is always True)
        if not self.trained:
            print('Model is not trained')
            return

        eval_metric = self.info['clf'].scoring
        est_name = self.info['clf'].estimator.__class__.__name__
        metric_names = {'Model is trained': self.trained,
                        'Best estimator': est_name,
                        'Evaluation metric': eval_metric,
                        'Dataset size': self.info["dataset_size"],
                        '# of features': self.info["feature_size"],
                        'Estimator set level': self.level,
                        'Including estimators': list(map(lambda x: x[0],
                                                     self.est_list)),
                        'Training time': "{:.2f} sec.".format(
                            self.info["training_time"])}

        features_df = pd.DataFrame(list(metric_names.items()),
                                   columns=['metric',
                                            'value']).set_index('metric')

        clf_metrics = ["Mean abs error",
                       "Mean sq error",
                       "r2",
                       eval_metric]

        metrics_df = pd.DataFrame({'metric': clf_metrics}).set_index('metric')
        train_eval_df = self.info["train_eval_df"]
        test_eval_df = self.info["test_eval_df"]

        eval_test = self.info['eval_df']['metric_test'].mean()
        eval_train = self.info['eval_df']['metric_train'].mean()
        abs_test = test_eval_df['neg_mean_absolute_error'].mean()
        abs_train = train_eval_df["neg_mean_absolute_error"].mean()
        sq_test = test_eval_df["neg_mean_squared_error"].mean()
        sq_train = train_eval_df["neg_mean_squared_error"].mean()
        r2_test = test_eval_df["r2"].mean()
        r2_train = train_eval_df["r2"].mean()

        metrics_df['training set'] = [abs_train,
                                      sq_train,
                                      r2_train,
                                      eval_train]
        metrics_df['test set'] = [abs_test,
                                  sq_test,
                                  r2_test,
                                  eval_test]
        if tab:
            print(tabulate(features_df, headers='keys', tablefmt='psql'))
            print(tabulate(metrics_df, headers='keys', tablefmt='psql'))
        else:
            features_df.head()
            metrics_df.head()

        return features_df, metrics_df
