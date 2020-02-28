import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import auc, roc_curve, log_loss, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

from learnit.autolearn.evaluate import Evaluate, MetricCatalog


def run_validation(X, y,
                   est,
                   metric,
                   cv_num=5,
                   validation_ratio=0.2,
                   pos_label=1,
                   n_jobs=1,
                   verbose=0):
    """Run Validation Meta

    Args:
        X (np.array): Feature matrix
        y (np.array): Label vector
        est either:
            (sklearn.base.ClassifierMixin): Classifier object
            (sklearn.base.RegressorMixin): Regressor object
        metric (str) : Evaluation metric
                       metric in ['roc_auc', 'neg_log_loss']
        cv_num (int): Number of fold for cross validation
        pos_label (int): Positive label name (used for binary classification)
        n_jobs (int): The number of jobs to run parallel processes
        verbose (int): Controls the verbosity

    """
    # this is the stub for regression. To be deleted after.
    assert type(n_jobs) == int
    if hasattr(est, 'n_jobs'):
        if hasattr(est, 'estimator'):
            if 'XGB' in est.estimator.__class__.__name__:
                # Set n_jobs to directly to XGB* estimator
                # for multi-thread instead of multi-process
                est.estimator.n_jobs = n_jobs
                # Then, set 1 to the number of parallelism of GridSearchCV
                est.n_jobs = 1
            else:
                est.n_jobs = n_jobs
        else:
            est.n_jobs = n_jobs
    if metric not in ['roc_auc', 'neg_log_loss', 'neg_mean_absolute_error']:
        raise ValueError(
            'run_validation received the unknown metric %s' % metric)

    if cv_num > 1:

        return __run_cross_validation(X, y, est, metric, cv_num,
                                      pos_label, verbose)
    else:
        if metric == 'neg_mean_absolute_error':
            return __reg_single_validation(X, y, est, validation_ratio,
                                           verbose=verbose)
        return __run_single_validation(
            X, y, est, metric, validation_ratio, pos_label, verbose
        )


def __run_cross_validation(X, y,
                           clf,
                           metric,
                           cv_num=5,
                           pos_label=1,
                           verbose=0):
    """Run cross validation for evaluation.

    Args:
        X (np.array): Feature matrix
        y (np.array): Label vector
        clf (sklearn.base.ClassifierMixin): Classifier object
        metric (str) : Evaluation metric
                       metric in ['roc_auc', 'neg_log_loss']
        cv_num (int): Number of fold for cross validation
        pos_label (int): Positive label name (used for binary classification)
        verbose (int): Controls the verbosity

    Returns:
        {'cv_df': pd.DataFrame(data_list,
                               columns=['metric_test',
                                        'metric_train']),
            'y_error': y_error,
            'y_pred': y_pred,
            'sample_clf': clf}

    """

    if metric == 'neg_mean_absolute_error':
        # regression task: using separate function
        return __reg_cross_validation(X=X, y=y, est=clf, cv_num=cv_num,
                                      verbose=verbose)

    data_list = []
    num_class = len(np.unique(y))
    kf = StratifiedKFold(n_splits=cv_num,
                         random_state=1)  # TODO(Yoshi): random_state

    # accuracy, precision, recall
    metric_func_dict = MetricCatalog.get_basic_metrics()
    train_eval_s_list = []
    test_eval_s_list = []

    # TODO(Yoshi): If clf (e.g., GridSearchCV) has inner classifier object
    # that has `verbose` paramter, the below logic does not handle it.
    assert type(verbose) == int
    if hasattr(clf, 'verbose'):
        clf.verbose = verbose

    if num_class > 2:
        y_error = np.zeros((len(y), num_class))
        y_pred_all = np.zeros((len(y), num_class))
    else:
        y_error = np.zeros(len(y))
        y_pred_all = np.zeros(len(y))

    for train_idx, test_idx in kf.split(X, y):
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test = X[train_idx], X[test_idx]
        clf.fit(X_train, y_train)

        # Take out class information from estimator or GridSearch object
        if hasattr(clf, 'classes_'):
            classes_ = clf.classes_
        else:
            assert hasattr(clf.best_estimator_, 'classes_')
            classes_ = clf.best_estimator_.classes_

        if not hasattr(clf, 'predict_proba'):
            clf = CalibratedClassifierCV(clf, cv='prefit')
            clf.fit(X_train, y_train)

        # predict/predict_proba
        if metric in ['roc_auc']:
            assert num_class == 2

            # Binary classification
            y_pred = clf.predict(X_test)
            pos_idx = np.where(np.array(classes_) == pos_label)[0][0]
            y_prob = clf.predict_proba(X_test)[:, pos_idx]
            y_pred_train = clf.predict(X_train)
            y_prob_train = clf.predict_proba(X_train)[:, pos_idx]
            y_error[test_idx] = np.abs(y_test - y_prob)
            y_pred_all[test_idx] = y_prob

            # Calculate evaulation metric
            fpr_test, tpr_test, _ = roc_curve(y_test,
                                              y_prob,
                                              pos_label=pos_label)
            metric_test = auc(fpr_test, tpr_test)
            fpr_train, tpr_train, _ = roc_curve(y_train,
                                                y_prob_train,
                                                pos_label=pos_label)
            metric_train = auc(fpr_train, tpr_train)
            train_eval_s = Evaluate.run_metric_functions(y_train,
                                                         y_pred_train,
                                                         y_prob_train,
                                                         metric_func_dict,
                                                         "binary")
            train_eval_s_list.append(train_eval_s)
            test_eval_s = Evaluate.run_metric_functions(y_test,
                                                        y_pred,
                                                        y_prob,
                                                        metric_func_dict,
                                                        "binary")
            test_eval_s_list.append(test_eval_s)

        elif metric in ['neg_log_loss']:
            print("metric in [neg_log_loss]")
            # Multi-class classification - we should not run it with binary!
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)  # matrix
            y_pred_train = clf.predict(X_train)
            y_prob_train = clf.predict_proba(X_train)  # matrix

            y_pred_all[test_idx] = y_prob

            # TODO(Yoshi): Cannot simply define y_error for multi
            y_error[test_idx] = np.nan

            print("Evaluate neg_log_loss")

            # Calculate evaluation metric.
            # Add the negative sign to make it a "score"
            metric_test = - log_loss(y_test, y_prob)
            metric_train = - log_loss(y_train, y_prob_train)

            train_eval_s = Evaluate.run_metric_functions(y_train,
                                                         y_pred_train,
                                                         y_prob_train,
                                                         metric_func_dict,
                                                         "multi")
            train_eval_s_list.append(train_eval_s)
            test_eval_s = Evaluate.run_metric_functions(y_test,
                                                        y_pred,
                                                        y_prob,
                                                        metric_func_dict,
                                                        "multi")
            test_eval_s_list.append(test_eval_s)

        else:
            raise Exception("Metric not supported: {}".format(metric))

        data_list.append([metric_test,
                          metric_train])

    return {'cv_df': pd.DataFrame(data_list,
                                  columns=['metric_test',
                                           'metric_train']),
            'train_eval_df': pd.concat(train_eval_s_list, axis=1).T,
            'test_eval_df': pd.concat(test_eval_s_list, axis=1).T,
            'y_error': y_error,
            'y_pred': y_pred,
            'sample_clf': clf}


# TODO(Bublin): Cut out common process between cross_validation and validation
def __run_single_validation(X, y,
                            clf,
                            metric,
                            validation_ratio=0.2,
                            pos_label=1,
                            verbose=0):
    """Run validation for evaluation.

    Args:
        X (np.array): Feature matrix
        y (np.array): Label vector
        clf (sklearn.base.ClassifierMixin): Classifier object
        metric (str) : Evaluation metric
                       metric in ['roc_auc', 'neg_log_loss']
        validation_ratio (float): size of validation data
        pos_label (int): Positive label name (used for binary classification)
        verbose (int): Controls the verbosity

    Returns:
        {'cv_df': pd.DataFrame(data_list,
                              columns=['metric_test',
                                       'metric_train']),
            'y_error': y_error,
            'y_pred': y_pred,
            'sample_clf': clf}

    """

    # TODO(Yoshi): Overall function should be able to merge with
    # run_cross_validation()

    data_list = []
    metric_func_dict = MetricCatalog.get_basic_metrics()
    train_eval_s_list = []
    test_eval_s_list = []
    num_class = len(np.unique(y))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio,
                                 random_state=0)
    train_idx, test_idx = next(sss.split(X, y))
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = X[train_idx], X[test_idx]
    clf.fit(X_train, y_train)
    # predict/predict_proba
    if metric in ['roc_auc']:
        assert num_class == 2
        # Take out class information from estimator or GridSearch object
        if hasattr(clf, 'classes_'):
            classes_ = clf.classes_
        else:
            assert hasattr(clf.best_estimator_, 'classes_')
            classes_ = clf.best_estimator_.classes_
        # Binary classification
        pos_idx = np.where(np.array(classes_) == pos_label)[0][0]
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, pos_idx]
        y_pred_train = clf.predict(X_train)
        y_prob_train = clf.predict_proba(X_train)[:, pos_idx]
        y_error = np.abs(y_test - y_pred)
        # Calculate evaulation metric
        fpr_test, tpr_test, _ = roc_curve(y_test,
                                          y_prob,
                                          pos_label=pos_label)
        metric_test = auc(fpr_test, tpr_test)
        fpr_train, tpr_train, _ = roc_curve(y_train,
                                            y_prob_train,
                                            pos_label=pos_label)
        metric_train = auc(fpr_train, tpr_train)

        train_eval_s = Evaluate.run_metric_functions(y_train,
                                                     y_pred_train,
                                                     y_prob_train,
                                                     metric_func_dict,
                                                     "binary")
        train_eval_s_list.append(train_eval_s)
        test_eval_s = Evaluate.run_metric_functions(y_test,
                                                    y_pred,
                                                    y_prob,
                                                    metric_func_dict,
                                                    "binary")
        test_eval_s_list.append(test_eval_s)
    elif metric in ['neg_log_loss']:
        print("metric in [neg_log_loss]")
        # Multi-class classification - we should not run it with binary!
        # TODO(Bublin): This y_pred don't have collect index
        # (do we have to return y_pred and y_error?)
        y_pred = clf.predict(X_test)  # matrix
        y_prob = clf.predict_proba(X_test)  # matrix
        y_pred_train = clf.predict(X_train)
        y_prob_train = clf.predict_proba(X_train)  # matrix
        # TODO(Yoshi): Cannot simply define y_error for multi
        y_error = np.nan
        print("Evaluate neg_log_loss")
        # Calculate evaluation metric
        metric_test = log_loss(y_test, y_prob)
        metric_train = log_loss(y_train, y_prob_train)

        train_eval_s = Evaluate.run_metric_functions(y_train,
                                                     y_pred_train,
                                                     y_prob_train,
                                                     metric_func_dict,
                                                     "multi")
        train_eval_s_list.append(train_eval_s)
        test_eval_s = Evaluate.run_metric_functions(y_test,
                                                    y_pred,
                                                    y_prob,
                                                    metric_func_dict,
                                                    "multi")
        test_eval_s_list.append(test_eval_s)
    else:
        raise Exception("Metric not supported: {}".format(metric))
    data_list.append([metric_test,
                      metric_train])

    return {'cv_df': pd.DataFrame(data_list,
                                  columns=['metric_test',
                                           'metric_train']),
            'train_eval_df': pd.concat(train_eval_s_list, axis=1).T,
            'test_eval_df': pd.concat(test_eval_s_list, axis=1).T,
            'y_error': y_error,
            'y_pred': y_pred,
            'test_index': test_idx,
            'sample_clf': clf}


def __reg_cross_validation(X, y, est, cv_num=5, n_jobs=1, verbose=0):
    """Cross validation for regression case

    Args:
        X (np.array): Feature matrix
        y (np.array): Label vector
        est (sklearn.base.Regressor): Regressor object
        cv_num (int): Number of fold for cross validation
        n_jobs (int): The number of jobs to run parallel processes
        verbose (int): Controls the verbosity

    Returns:
        {'cv_df': pd.DataFrame(data_list,
                               columns=['metric_test',
                                        'metric_train'])
        'train_eval_df':
        'test_eval_df' :
        sample_est: est
        }

    """

    data_list = []
    train_eval_s_list = []
    test_eval_s_list = []
    metric_func_dict = MetricCatalog.get_basic_metrics(task_type='regression')

    kf = KFold(n_splits=cv_num, random_state=1)  # TODO(Yoshi): random_state

    assert type(verbose) == int
    if hasattr(est, 'verbose'):
        est.verbose = verbose

    if hasattr(est, 'n_jobs'):
        est.n_jobs = n_jobs

    for train_idx, test_idx in kf.split(X, y):

        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test = X[train_idx], X[test_idx]
        est.fit(X_train, y_train)

        y_pred = est.predict(X_test)
        y_pred_train = est.predict(X_train)
        metric_test = - mean_absolute_error(y_test, y_pred)
        metric_train = - mean_absolute_error(y_train, y_pred_train)

        data_list.append([metric_test,
                          metric_train])
        train_eval_s = Evaluate.run_metric_functions(y_train,
                                                     y_pred_train,
                                                     None,
                                                     metric_func_dict,
                                                     "regression")
        train_eval_s_list.append(train_eval_s)

        test_eval_s = Evaluate.run_metric_functions(y_test,
                                                    y_pred,
                                                    None,
                                                    metric_func_dict,
                                                    "regression")
        test_eval_s_list.append(test_eval_s)

    return {'cv_df': pd.DataFrame(data_list, columns=['metric_test',
                                                      'metric_train']),
            'train_eval_df': pd.concat(train_eval_s_list, axis=1).T,
            'test_eval_df': pd.concat(test_eval_s_list, axis=1).T,
            'sample_est': est}


def __reg_single_validation(X, y,
                            est,
                            validation_ratio=0.2,
                            verbose=0):
    """Run single validation for regression.

    Args:
        X (np.array): Feature matrix
        y (np.array): Label vector
        est (sklearn.base.RegressorMixin): Regressor object
        validation_ratio (float): size of validation data
        n_jobs (int): The number of jobs to run parallel processes
        verbose (int): Controls the verbosity

    Returns:
        {'cv_df': pd.DataFrame(data_list,
                              columns=['metric_test',
                                       'metric_train']),
            'train_eval_df': None,
            'test_eval_df': None,
            'sample_est': est}

    """

    if type(verbose) != int:
        raise ValueError('Verbose parameter must be an integer')

    if hasattr(est, 'verbose'):
        est.verbose = verbose

    data_list = []
    train_eval_s_list = []
    test_eval_s_list = []
    metric_func_dict = MetricCatalog.get_basic_metrics(task_type='regression')

    ss = ShuffleSplit(n_splits=1, test_size=validation_ratio,
                      random_state=0)
    train_idx, test_idx = next(ss.split(X, y))
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = X[train_idx], X[test_idx]
    est.fit(X_train, y_train)

    y_pred = est.predict(X_test)  # matrix
    y_pred_train = est.predict(X_train)

    metric_test = mean_absolute_error(y_test, y_pred)
    metric_train = mean_absolute_error(y_train, y_pred_train)

    data_list.append([metric_test,
                      metric_train])
    train_eval_s = Evaluate.run_metric_functions(y_train,
                                                 y_pred_train,
                                                 None,
                                                 metric_func_dict,
                                                 task_type="regression")
    train_eval_s_list.append(train_eval_s)
    test_eval_s = Evaluate.run_metric_functions(y_test,
                                                y_pred,
                                                None,
                                                metric_func_dict,
                                                task_type="regression")
    test_eval_s_list.append(test_eval_s)

    return {'cv_df': pd.DataFrame(data_list,
                                  columns=['metric_test',
                                           'metric_train']),
            'train_eval_df': pd.concat(train_eval_s_list, axis=1).T,
            'test_eval_df': pd.concat(test_eval_s_list, axis=1).T,
            'sample_est': est}
