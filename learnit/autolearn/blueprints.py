import random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVR


try:
    from xgboost import XGBClassifier
    GBClassifier = XGBClassifier
except ImportError:
    print(("WARNING: xgboost not installed. ",
           "Use sklearn.ensemble.GradientBoostingClassifier instead."))
    from sklearn.ensemble import GradientBoostingClassifier
    GBClassifier = GradientBoostingClassifier


class ParamCatalog:
    # TODO(Yoshi): This will be an external JSON file or somehting like that
    param_sets = {'grid':
                  {'LogisticRegression': [
                      {'C': [0.001, 0.1, 1.0, 10.0]}],
                   'LinearSVC': [
                      {'C': [0.001, 0.1, 1.0, 10.0]}],
                   'KNeighborsClassifier': [
                      {'n_neighbors': [1, 3, 5, 7]}],
                   'XGBClassifier': [
                       {'n_estimators': [10, 50],
                        'learning_rate': [0.01, 0.05],
                        'max_depth': [2, 5],
                        'subsample': [0.75],
                        'colsample_bytree': [0.8]}],
                   'GradientBoostingClassifier': [
                       {'n_estimators': [10, 50],
                        'learning_rate': [0.01, 0.05],
                        'max_depth': [2, 5],
                        'subsample': [0.75],
                        'max_features': [0.8]}],
                   'RandomForestClassifier': [
                       {'n_estimators': [30, 50, 100],
                        'max_features': [0.6, 0.8],
                        'max_depth': [5, 10, 20]}],
                   'Ridge': [
                       {'alpha': [0, 0.5, 1],
                        'tol': [0.0001, 0.01, 0.1]}],
                   'SVR': [
                       {'C': [0.001, 0.1, 1.0, 10.0, 50.0, 100.0],
                        'epsilon': [0.1, 0.05],
                        'kernel': ['rbf']}]},
                  'random': {}
                  }

    @classmethod
    def get_params(cls, clf_name, search_method='grid', choice_idx=None):
        """Returns parameter set predefined in the class

        Args:
            clf_name:
            search_method (str): 'grid' or 'random'
            choice_idx:

        Returns:

        """
        assert search_method in cls.param_sets
        assert clf_name in cls.param_sets[search_method]

        if choice_idx is None:
            return random.choice(cls.param_sets[search_method][clf_name])
        else:
            assert choice_idx >= len(cls.param_sets[search_method][clf_name])
            return cls.param_sets[search_method][clf_name][choice_idx]


class PredictProbaFeature(BaseEstimator, TransformerMixin):
    """Returns clf.predict_proba() as features."""

    def __init__(self, clf, pos_label=1):
        """Init function

        Args:
            clf: Classifier
            pos_label: positive label (default=1)

        """
        assert hasattr(clf, 'predict_proba')
        self.estimator_ = clf
        self.pos_label_ = pos_label

    def fit(self, X, y=None):
        """fit

        Args:
            X: Data
            y: label

        Returns:
            self

        """

        self.estimator_.fit(X, y)

        if self.pos_label_ is not None:
            if hasattr(self.estimator_, 'classess_'):
                self.pos_idx_ = np.where(
                    np.array(
                        self.estimator_.classes_) == self.pos_label_)[0][0]
            else:
                # CV classes such as GridSearchCV, RandomSearchCV do not
                # implement classess_
                assert hasattr(self.estimator_, 'best_estimator_')
                classes = self.estimator_.best_estimator_.classes_
                self.pos_idx_ = np.where(
                    np.array(classes) == self.pos_label_)[0][0]
        return self

    def transform(self, X):
        """Returns probabilities for all classess if pos_label_ is None

        Args:
            X: Data

        """

        if self.pos_label_ is None:
            return self.estimator_.predict_proba(X)
        else:
            return self.estimator_.predict_proba(X)[:, self.pos_idx_][None].T


class PrefitTransformer(BaseEstimator, TransformerMixin):
    """Avoid run fit() method when transformers are pre-fitted."""

    def __init__(self, transformer):
        self.transformer_ = transformer

    def fit(self, X, y=None):
        # Do nothing
        return self

    def transform(self, X):
        return self.transformer_.transform(X)


class DummyTransformer(BaseEstimator, TransformerMixin):
    """Returns original features as they are."""

    def fit(self, X, y=None):
        # Do nothing
        return self

    def transform(self, X):
        return X


class AverageBlenderClassifier(BaseEstimator, ClassifierMixin):
    """Returns average."""

    def __init__(self, coef=None, scaler=StandardScaler()):
        """Init function

        Args:
            coef:
            scaler:

        """

        self.coef_ = coef
        self.scaler_ = scaler
        self.b = 0.0
        if scaler.__class__.__name__ == 'MinMaxScaler':
            self.b = 0.5
        self.classes_ = np.array([1, 0])

    def fit(self, X, y=None):
        if self.coef_ is None:
            _, M = X.shape
            self.coef_ = np.ones(M) / M
        self.scaler_.fit(np.dot(X, self.coef_).reshape(-1, 1))
        return self

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(- x + self.b))

    def predict_proba(self, X):
        probs = self._sigmoid(
            self.scaler_.transform(np.dot(X, self.coef_)[None].T))
        return np.c_[(probs, 1.0 - probs)]


class GridSearchCVFactory:
    """Returns GridSearchCV instance with input estimator and parameters."""

    @classmethod
    def create(cls,
               clf,
               scoring,
               param_catalog=None,
               cv=5,
               n_jobs=1,
               verbose=3,
               choice_idx=None):
        """create

        Args:
            clf: sklearn estimator
            scoring (str): scoring metric name (e.g., 'roc_auc')
            param_catalog (class): ParamCatalog will be used if None
            cv (int): number of cross validation iteration
            n_jobs (int):
            verbose (int):
            choice_idx (int): parameter set idx. Choose random if None
                              (Default: None)

        Returns: GridSearchCV object

        """

        if param_catalog is None:
            param_catalog = ParamCatalog

        return GridSearchCV(clf,
                            param_catalog.get_params(clf.__class__.__name__,
                                                     'grid',
                                                     choice_idx=choice_idx),
                            scoring=scoring,
                            cv=cv,
                            n_jobs=n_jobs,
                            verbose=verbose)


class AverageBlender:
    """AverageBlender class."""
    def __init__(self,
                 scoring,
                 random_state=1,
                 verbose=0,
                 pos_label=1,
                 param_catalog=ParamCatalog):
        """Init function

        Args:
            scoring:
            random_state:
            verbose:
            pos_label:
            param_catalog:

        """
        print(scoring)
        random.seed(random_state)
        np.random.seed(random_state)
        self.random_state_ = random_state
        self.scoring_ = scoring
        transformer_list = [
            ('1', PredictProbaFeature(
                GridSearchCVFactory.create(param_catalog,
                                           LogisticRegression(),
                                           scoring=scoring,
                                           verbose=verbose),
                pos_label=pos_label)),
            ('2', PredictProbaFeature(
                GridSearchCVFactory.create(param_catalog,
                                           GBClassifier(seed=1),
                                           scoring=scoring,
                                           verbose=verbose),
                pos_label=pos_label)),
            ('3', PredictProbaFeature(
                GridSearchCVFactory.create(param_catalog,
                                           GBClassifier(seed=2),
                                           scoring=scoring,
                                           verbose=verbose),
                pos_label=pos_label)),
            ('4', PredictProbaFeature(
                GridSearchCVFactory.create(param_catalog,
                                           GBClassifier(seed=3),
                                           scoring=scoring,
                                           verbose=verbose),
                pos_label=pos_label)),
            ('5', PredictProbaFeature(
                GridSearchCVFactory.create(param_catalog,
                                           RandomForestClassifier(
                                               random_state=1),
                                           scoring=scoring,
                                           verbose=verbose),
                pos_label=pos_label))]
        self.fu1 = FeatureUnion(transformer_list=transformer_list)
        self.pipe = Pipeline([('1', self.fu1),
                              ('2', AverageBlenderClassifier())])

    def fit(self, X, y):
        self.pipe.fit(X, y)
        self.classes_ = self.pipe.steps[-1][1].classes_
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)

    def predict_proba_class(self, X, label=1):
        classes = self.pipe.steps[-1][1].best_estimator_.classes_
        idx = np.where(np.array(classes) == label)[0][0]
        return self.predict_proba(X)[:, idx]


class StackedXGBoost:
    """Stacked XGBoost classifier."""
    def __init__(self,
                 scoring,
                 random_state=1,
                 layer_ratios=[0.33, 0.33],
                 verbose=0,
                 pos_label=1,
                 param_catalog=ParamCatalog):
        """Init function

        Args:
            scoring:
            random_state:
            layer_ratios:
            verbose:
            pos_label:
            param_catalog:

        """
        random.seed(random_state)
        np.random.seed(random_state)
        self.random_state_ = random_state
        self.scoring_ = scoring
        self.layer_ratios_ = layer_ratios
        self.pos_label_ = pos_label

        transformer_list1 = [
            ('1', Pipeline([('1', PredictProbaFeature(
                GridSearchCVFactory.create(param_catalog,
                                           LogisticRegression(),
                                           scoring=scoring,
                                           verbose=verbose),
                pos_label=pos_label))])),
            ('2', Pipeline([
                ('1', TruncatedSVD(n_components=5)),
                ('2', PredictProbaFeature(
                    GridSearchCVFactory.create(param_catalog,
                                               LogisticRegression(),
                                               scoring=scoring,
                                               verbose=verbose),
                    pos_label=pos_label))])),
            ('3', Pipeline([
                ('1', NMF(n_components=10)),
                ('2', PredictProbaFeature(
                    GridSearchCVFactory.create(param_catalog,
                                               LogisticRegression(),
                                               scoring=scoring,
                                               verbose=verbose),
                    pos_label=pos_label))]))]

        self.fu1 = FeatureUnion(transformer_list=transformer_list1)

        transformer_list2 = [
            ('1', Pipeline([
                ('1', FeatureUnion(transformer_list=[
                    ('1', PrefitTransformer(self.fu1)),
                    ('2', DummyTransformer())])),
                ('2', PredictProbaFeature(
                    GridSearchCVFactory.create(param_catalog,
                                               GBClassifier(),
                                               scoring=scoring,
                                               verbose=verbose),
                    pos_label=pos_label))])),
            ('2', Pipeline([
                ('1', FeatureUnion(transformer_list=[
                    ('1', PrefitTransformer(self.fu1)),
                    ('2', DummyTransformer())])),
                ('2', PredictProbaFeature(
                    GridSearchCVFactory.create(param_catalog,
                                               GBClassifier(),
                                               scoring=scoring,
                                               verbose=verbose),
                    pos_label=pos_label))]))]

        self.fu2 = FeatureUnion(transformer_list=transformer_list2)
        self.pipe = Pipeline([
            ('1', PrefitTransformer(self.fu2)),
            ('2', GridSearchCVFactory.create(param_catalog,
                                             LogisticRegression(),
                                             scoring=scoring,
                                             verbose=verbose))])

    def fit(self, X, y):
        N, _ = X.shape
        idx1 = int(N * self.layer_ratios_[0])
        idx2 = int(N * (self.layer_ratios_[0] + self.layer_ratios_[1]))
        X1, y1 = X[:idx1, :], y[:idx1]
        X2, y2 = X[idx1:idx2, :], y[idx1:idx2]
        X3, y3 = X[idx2:, :], y[idx2:]
        self.fu1.fit(X1, y1)
        self.fu2.fit(X2, y2)
        self.pipe.fit(X3, y3)
        self.classes_ = self.pipe.steps[-1][1].best_estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)

    def predict_proba_class(self, X, label=1):
        classes = self.pipe.steps[-1][1].best_estimator_.classes_
        idx = np.where(np.array(classes) == label)[0][0]
        return self.predict_proba(X)[:, idx]


# Individual classifiers
LogisticRegressionGridSearchCV = GridSearchCVFactory.create(
    clf=LogisticRegression(),
    scoring="roc_auc",
    verbose=3)

LinearSVCGridSearchCV = GridSearchCVFactory.create(
    clf=LinearSVC(),
    scoring="roc_auc",
    verbose=3)

RandomForestClassifierGridSearchCV = GridSearchCVFactory.create(
    clf=RandomForestClassifier(),
    scoring="roc_auc",
    verbose=3)

KNeighborsClassifierGridSearchCV = GridSearchCVFactory.create(
    clf=KNeighborsClassifier(),
    scoring="roc_auc",
    verbose=3)

GBClassifierGridSearchCV = GridSearchCVFactory.create(
    clf=GBClassifier(),
    scoring="roc_auc",
    verbose=3)


class ClassifierCatalog:
    level1 = [('LogisticRegression', LogisticRegressionGridSearchCV)]
    level2 = [('GBClassifier', GBClassifierGridSearchCV)]
    level3 = level1 + level2
    level4 = [('LogisticRegression', LogisticRegressionGridSearchCV),
              ('LinearSVC', LinearSVCGridSearchCV),
              ('RandomForestClassifierGridSearchCV',
               RandomForestClassifierGridSearchCV),
              ('KNeighborsClassifier',
               KNeighborsClassifierGridSearchCV)]
    """
    level5 = level3 + level4 + [('StackedXGBoost',
                                 StackedXGBoost(scoring="roc_auc")),
                                ('AverageBlender',
                                 AverageBlender(scoring="roc_auc"))]
    """

# Individual classifiers for multi-class classification
# TODO(Yoshi): These templates could be combined with binary classifiers
MultiLogisticRegressionGridSearchCV = GridSearchCVFactory.create(
    clf=LogisticRegression(),
    scoring="neg_log_loss",
    verbose=3)

MultiRandomForestClassifierGridSearchCV = GridSearchCVFactory.create(
    clf=RandomForestClassifier(),
    scoring="neg_log_loss",
    verbose=3)

MultiKNeighborsClassifierGridSearchCV = GridSearchCVFactory.create(
    clf=KNeighborsClassifier(),
    scoring="neg_log_loss",
    verbose=3)

MultiGBClassifierGridSearchCV = GridSearchCVFactory.create(
    clf=GBClassifier(),
    scoring="neg_log_loss",
    verbose=3)

RidgeRegressionGridSearchCV = GridSearchCVFactory.create(
    clf=Ridge(),
    scoring="neg_mean_absolute_error",
    verbose=3)

SVRRegressionGridSearchCV = GridSearchCVFactory.create(
    clf=SVR(),
    scoring="neg_mean_absolute_error",
    verbose=3)


class MultiClassifierCatalog:
    level1 = [('MultiLogisticRegression', MultiLogisticRegressionGridSearchCV)]
    level2 = [('MultiGBClassifier', MultiGBClassifierGridSearchCV)]
    level3 = level1 + level2
    level4 = [('MultiLogisticRegression', MultiLogisticRegressionGridSearchCV),
              ('MultiRandomForestClassifierGridSearchCV',
               MultiRandomForestClassifierGridSearchCV),
              ('MultiKNeighborsClassifier',
               MultiKNeighborsClassifierGridSearchCV)]


class RegressorCatalog:
    level1 = [('Ridge', RidgeRegressionGridSearchCV)]
    level2 = level1 + [('SVR', SVRRegressionGridSearchCV)]
