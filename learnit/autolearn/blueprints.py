import random

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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


class ClassifierCatalog:
    level1 = ["libsvm_svc", "sgd"]
    level2 = level1 + ["random_forest", "gradient_boosting"]
    level3 = None

# Individual classifiers for multi-class classification


class MultiClassifierCatalog:
    level1 = ["libsvm_svc", "sgd"]
    level2 = level1 + ["random_forest", "gradient_boosting"]
    level3 = None


class RegressorCatalog:
    level1 = ["ridge_regression", "liblinear_svr"]
    level2 = level1 + ["random_forest", "gradient_boosting"]
    level3 = None