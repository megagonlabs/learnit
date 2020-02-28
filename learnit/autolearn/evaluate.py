import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics


class MetricCatalog:
    catalog_dict = {
        'accuracy': {
            'func': metrics.accuracy_score,
            'params': {},
            'require_score': False,
            'binary': True,
            'multi': True},
        # AP is not straightfoward to apply to multiclass
        'average_precision': {
            'func': metrics.average_precision_score,
            'params': {},
            'require_score': True,
            'binary': True,
            'multi': False},
        # Default configuration only handles binary classification
        'f1': {
            'func': metrics.f1_score,
            'params': {'average': 'macro'},
            'require_score': False,
            'binary': True,
            'multi': True},
        'f1_micro': {
            'func': metrics.f1_score,
            'params': {'average': 'micro'},
            'require_score': False,
            'binary': True,
            'multi': True},
        'f1_macro': {
            'func': metrics.f1_score,
            'params': {'average': 'macro'},
            'require_score': False,
            'binary': True,
            'multi': True},
        # Note: log_loss returns "loss" value
        'neg_log_loss': {
            'func': lambda y_true, y_pred: - metrics.log_loss(y_true, y_pred),
            'params': {},
            'require_score': True,
            'binary': True,
            'multi': True},
        # Same problem as f1_score
        'precision': {
            'func': metrics.precision_score,
            'params': {'average': 'macro'},
            'require_score': False,
            'binary': True,
            'multi': True},
        'precision_micro': {
            'func': metrics.precision_score,
            'params': {'average': 'micro'},
            'require_score': False,
            'binary': True,
            'multi': True},
        'precision_macro': {
            'func': metrics.precision_score,
            'params': {'average': 'macro'},
            'require_score': False,
            'binary': True,
            'multi': True},
        # Same problem as f1_score
        'recall': {
            'func': metrics.recall_score,
            'params': {'average': 'macro'},
            'require_score': False,
            'binary': True,
            'multi': True},
        'recall_micro': {
            'func': metrics.recall_score,
            'params': {'average': 'micro'},
            'require_score': False,
            'binary': True,
            'multi': True},
        'recall_macro': {
            'func': metrics.recall_score,
            'params': {'average': 'macro'},
            'require_score': False,
            'binary': True,
            'multi': True},
        'roc_auc': {
            'func': metrics.roc_auc_score,
            'params': {},
            'require_score': True,
            'binary': True,
            'multi': False},
        # Regression metrics
        'explained_variance': {
            'func': metrics.explained_variance_score,
            'params': {},
            'require_score': False,
            'regression': True},
        'neg_mean_absolute_error': {
            'func': lambda y_true, y_pred: - metrics.mean_absolute_error(
                y_true, y_pred),
            'params': {},
            'require_score': False,
            'regression': True},
        'neg_mean_squared_error': {
            'func': lambda y_true, y_pred: - metrics.mean_squared_error(
                y_true, y_pred),
            'params': {},
            'require_score': False,
            'regression': True},
        'neg_median_absolute_error': {
            'func': lambda y_true, y_pred: - metrics.median_absolute_error(
                y_true, y_pred),
            'params': {},
            'require_score': False,
            'regression': True},
        'r2': {
            'func': metrics.r2_score,
            'params': {},
            'require_score': False,
            'regression': True}}

    @classmethod
    def get_basic_metrics(cls,
                          task_type="classification"):
        if task_type in ["classification",
                         "binary",
                         "multi"]:
            return dict(
                filter(lambda x: x[0] in ["accuracy",
                                          "precision",
                                          "recall"],
                       cls.catalog_dict.items()))

        elif task_type in ["regression", "reg"]:
            return dict(
                filter(lambda x: x[0] in ["neg_mean_absolute_error",
                                          "neg_mean_squared_error",
                                          "r2"],
                       cls.catalog_dict.items()))


class ErrorSummary(object):
    """Error Analysis summary class."""
    def __init__(self,
                 error_dist=None,
                 diversity=None,
                 errors=None):
        """Initialization

        Args:
            error_dist (pd.DataFrame): Error distribution table
            diversity (pd.DataFrame): Diversity metric table
            errors (pd.DataFrame): Misclassified examples

        """
        self.error_dist = error_dist
        self.diversity = diversity
        self.errors = errors


class Evaluate():
    def __init__(self,
                 alearn,
                 ac=None,
                 feature_names=None,
                 random_state=7):
        """Data evaluation class

        Args:
            alearn (AutoLearn or sklearn classifier instance):
                Trained model instance
            ac (AutoConverter instance): Autoconverter for converting column
                data to feature matrix
            feature_names (list): List of feature names (str)
                If ac is given, the parameter will be disregarded.
                If not, feature_names becomes mandatory.
            random_state (int): random seed for pandas.sample. Default: 7

        """

        if ac is None:
            if feature_names is None:
                raise ValueError("Either AutoConverter or feature_names must",
                                 "be given.")
            self.feature_names = feature_names
            self.ac = None
        else:
            self.ac = ac
            if feature_names is not None:
                warnings.warn("AutoConverter instance is given so",
                              "feature_names will be discarded.")
                self.feature_names = None

        # TODO(Yoshi): Need to modify when it incorporates regression type
        assert hasattr(alearn, "predict")
        assert hasattr(alearn, "predict_proba")

        if alearn.__class__.__name__ == "AutoLearn":
            assert alearn.trained
        else:
            # scikit-learn classifiers do not have "fitted" flag
            # A solution would be calling predict()/predict_proba()
            # to see if it returns exception.
            pass

        self.alearn = alearn
        self.rs = random_state
        self.orig_eval_s = None

    def _task_type(self):
        """Extract task_type from alearn (could be sklearn clf) instance."""

        if hasattr(self.alearn, 'task'):
            # AutoLearn instance passed
            if self.alearn.task == 'regression':
                task_type = 'regression'
            elif hasattr(self.alearn.learner, "task_type"):
                task_type = self.alearn.learner.task_type
            else:
                raise ValueError("wrong task_type passed to evaluate")

        else:
            # in this case we have scikit-learn classifier passed
            if isinstance(self.alearn, sklearn.base.ClassifierMixin):
                if len(self.alearn.classes_) == 2:
                    task_type = "binary"
                else:
                    task_type = "multi"
            elif isinstance(self.alearn, sklearn.base.RegressorMixin):
                task_type = "regression"
            else:
                raise ValueError("Unknown instance type: {}".format(
                    type(self.alearn)))

        return task_type

    def _pos_label(self):
        if hasattr(self.alearn, "pos_label"):
            return self.alearn.pos_label
        else:
            # Assume that the second index is positive
            return 1

    def get_feature_indexes(self):
        """Returns di

        Returns:
            table_colname_pos_dict =
                {"main..Ticket": [0, 20], "main..Age": [21, 30], ...}

        """
        if self.ac is not None:
            all_feature_names = self.ac.feature_names
        else:
            all_feature_names = self.feature_names

        # table_feature_names_cols =
        #     ["main..Ticket", "main..Ticket", ...]
        table_feature_name_cols = list(map(
            lambda x: x.split('..')[0] + ".." + x.split('..')[1].split('.')[0],
            all_feature_names))

        table_colname_pos_dict = {}
        begin = 0
        table_colname = table_feature_name_cols[0]
        counter = 0

        for i, feature_name in enumerate(table_feature_name_cols):
            if feature_name == table_colname:
                counter += 1
            else:
                # end is not included to the interval
                table_colname_pos_dict[table_colname] = [begin, i]
                begin = i
                counter = 1
                table_colname = feature_name

        table_colname_pos_dict[table_colname] = [begin,
                                                 len(table_feature_name_cols)]
        return table_colname_pos_dict

    @classmethod
    def run_metric_functions(cls,
                             y,
                             y_pred,
                             y_prob,
                             metric_func_dict,
                             task_type):
        """Run metric functions

        Args:
            y (np.ndarray): True label vector
            y_pred (np.ndarray): Predicted label vector
            y_prob (np.ndarray): Probability vector
                                 None if task_type == "regression"
            metric_func_dict (dict): metric func dictionary
                                    see MetricCatalog for details
            task_type (str): task type {"binary", "multi", "regression"}

        Returns:
            orig_eval_s (pd.Series)

        """
        if task_type not in ["binary", "multi", "regression"]:
            raise ValueError('task_type must be {"binary", "multi",'
                             '"regression"}')

        if task_type == "regression" and y_prob is not None:
            warnings.warn("y_prob will be disregarded for"
                          "task_type=regression")

        # Only use evaluation metric that supports task_type
        sorted_metric_names = sorted(
            filter(lambda x: (task_type in metric_func_dict[x] and
                              metric_func_dict[x][task_type]),
                   metric_func_dict.keys()))
        # Evaluate prediction
        eval_list = []
        for metric_name in sorted_metric_names:
            metric_info = metric_func_dict[metric_name]
            metric_func = metric_info['func']
            metric_params = metric_info['params']
            assert metric_info[task_type]

            if metric_info["require_score"]:
                score = metric_func(y, y_prob, **metric_params)
            else:
                # Evaluation metrics for regression use y_pred
                score = metric_func(y, y_pred, **metric_params)

            eval_list.append(score)
        orig_eval_s = pd.Series(eval_list, index=sorted_metric_names)
        return orig_eval_s

    def evaluate_performance(self,
                             X=None,
                             y=None,
                             metric_func_dict=None):
        """Evaluate prediction performance.

        Args:
            df (pd.DataFrame): Main table
            X (np.array): Test feature matrix
            y (np.array): Test label vector
            metric_func_dict (dict): if None, it will use MetricCatalog
              {"metric_name": {"func": func,
                               "params": {},
                               "require_score": True,
                               "binary": True,
                               "multi": True}}

        Returns:
            orig_eval_s (pd.Series): Evaluation values

        """
        if metric_func_dict is None:
            metric_func_dict = MetricCatalog.catalog_dict

        if (X is None) or (y is None):
            if self.ac is None:
                raise ValueError(
                    "X and y are missing since AutoConverter instance was not",
                    "given.")
            if not self.ac.hasdata:
                raise RuntimeError(
                    "AutoConverter instance does not store X and y.")
            X = self.ac.X
            y = self.ac.y

        # 1. pure prediction
        y_pred = self.alearn.predict(X)
        if self._task_type() in ["binary", "multi"]:
            y_prob = self.alearn.predict_proba(X)
            if self._task_type() == "binary":
                y_prob = y_prob[:, self._pos_label()]
        else:
            # y_prob is empty for regression
            y_prob = None

        # y_pred, y_prob, metric_func_dict
        self.orig_eval_s = Evaluate.run_metric_functions(y,
                                                         y_pred,
                                                         y_prob,
                                                         metric_func_dict,
                                                         self._task_type())
        return self.orig_eval_s

    def calculate_column_importance(self,
                                    X=None,
                                    y=None,
                                    target=None,
                                    metric_func_dict=None):
        """Evaluate column importance scores

        Args:
            X (np.array): Test feature matrix
            y (np.array): Test label vector
            column_importance (bool): Calculate column importance if True
                                      Default=True,
            metric_func_dict (dict): if None, it will use MetricCatalog
              {"metric_name": {"func": func,
                               "params": {},
                               "require_score": True,
                               "binary": True,
                               "multi": True}}

        Returns:
            col_imp_df (pd.DataFrame):

                                    accuracy  average_precision        f1 ...
                tablename colname
                main      Age       0.012240           0.007844  0.013407 ...
                          Cabin     0.040392           0.024465  0.044803 ...
                          Embarked  0.008568           0.006306  0.009215 ...
                          Fare      0.009792           0.002827  0.010472 ...
                          Name      0.046512           0.057124  0.050983 ...
                          Parch     0.000000           0.000600  0.000127 ...
                          Pclass    0.029376           0.027463  0.031666 ...
                          Sex       0.227662           0.236873  0.244964 ...
                          SibSp     0.006120           0.006541  0.006973 ...
                          Ticket    0.055080           0.072796  0.058413 ...

        """
        if metric_func_dict is None:
            metric_func_dict = MetricCatalog.catalog_dict

        if (X is None) or (y is None):
            if self.ac is None:
                raise ValueError(
                    "X and y must be given since it has no AutoConverter",
                    "instance.")
            if not self.ac.hasdata:
                raise RuntimeError(
                    "AutoConverter instance does not store X and y.")
            X = self.ac.X
            y = self.ac.y

        if self.ac is None:
            if target is None:
                raise ValueError("target parameter must be given since",
                                 "it has no AutoConverter instance.")
        else:
            target = self.ac.target
            if target is not None:
                warnings.warn("Give target will be discarded.")

        if self.orig_eval_s is None:
            self.evaluate_performance(X=X,
                                      y=y,
                                      metric_func_dict=metric_func_dict)
        assert self.orig_eval_s is not None

        # feature_indexes_dict[table_colname] = [begin, end]
        feature_indexes_dict = self.get_feature_indexes()

        # Only use evaluation metric that supports task_type
        sorted_metric_names = sorted(
            filter(lambda x: (self._task_type() in metric_func_dict[x] and
                              metric_func_dict[x][self._task_type()]),
                   metric_func_dict.keys()))

        # Column importance
        col_importance_list = []
        col_imp_index_list = []

        for table_colname in sorted(feature_indexes_dict.keys()):
            tablename, colname = table_colname.split('..')
            if tablename == 'main' and colname == target:
                continue
            col_imp_index_list.append(table_colname)
            # Get needed feature columns range and spoil them
            beg_idx, end_idx = feature_indexes_dict[table_colname]
            X_shuf = X.copy()
            np.random.shuffle(X_shuf[:, beg_idx:end_idx])

            # Permuted prediction
            y_shuf_pred = self.alearn.predict(X_shuf)
            if self._task_type() in ["binary", "multi"]:
                y_shuf_prob = self.alearn.predict_proba(X_shuf)
                if self._task_type() == 'binary':
                    y_shuf_prob = y_shuf_prob[:, self._pos_label()]

            # Calculate evaluation
            metric_list = []
            for metric_name in sorted_metric_names:
                metric_info = metric_func_dict[metric_name]
                metric_func = metric_info['func']
                metric_params = metric_info['params']
                assert metric_info[self._task_type()]

                if metric_info["require_score"]:
                    # orig_score = metric_func(y, y_prob)
                    orig_score = self.orig_eval_s[metric_name]
                    shuf_score = metric_func(y, y_shuf_prob, **metric_params)
                else:
                    # orig_score = metric_func(y, y_pred)
                    orig_score = self.orig_eval_s[metric_name]
                    shuf_score = metric_func(y, y_shuf_pred, **metric_params)

                # TODO(Yoshi): Double check if there is no problem
                # for neg_log_loss
                if orig_score == 0:
                    metric_list.append(0.0)
                else:
                    metric_list.append((orig_score - shuf_score) / orig_score)
            col_importance_list.append(metric_list)

        col_imp_df = pd.DataFrame(col_importance_list)
        col_imp_df.columns = sorted_metric_names

        tablename_list = list(map(lambda x: x.split('..')[0],
                                  col_imp_index_list))
        colname_list = list(map(lambda x: x.split('..')[1],
                                col_imp_index_list))
        assert len(tablename_list) == len(col_imp_df)
        assert len(tablename_list) == len(colname_list)

        assert "tablename" not in sorted_metric_names
        assert "colname" not in sorted_metric_names

        col_imp_df["tablename"] = tablename_list
        col_imp_df["colname"] = colname_list
        col_imp_df.set_index(["tablename", "colname"], inplace=True)

        return col_imp_df

    def get_top_columns(self, n=3):
        """Returns n most important columns in the DataFrame

            Args:
                n (integer): number of columns returned

            Returns:
                list of [tablename..columname, ...] of most
                    important columns, sorted in descending order

        """

        col_imp_df = self.calculate_column_importance()

        if self._task_type() == 'binary':
            metric = 'roc_auc'
        else:
            metric = 'neg_log_loss'

        new_df = col_imp_df[metric].sort_values(ascending=False).head(n)
        return list(map(lambda x: x[0] + '..' + x[1], new_df.index.values))

    def get_mispredictions(self, df):
        """Get mispredicted examples based on the classifier

        Args:
            df (pd.DateFrame): dataset to evaluate.

        Returns:
            mispred_df (pd.DataFrame):

        TODO(Yoshi): subtable support

        """
        # Assume AutoConverter is mandatory for the function
        if self.ac is None:
            raise ValueError("AutoConverter instance is required to call",
                             "get_mispredictions()")

        # TODO(Yoshi): This is not accurate.
        # AutoConverter also should have "fitted" flag or something like that.
        assert self.ac.hasdata

        X, y = self.ac.transform(df)
        pred_y = self.alearn.predict(X)

        # TODO(Yoshi): Add some columns such as ==prediction== column,
        # ==confidence==. To be disccused and will be another ticket.
        return df.ix[y != pred_y]

    def stratify_errors(self,
                        df,
                        max_numcat=5):
        """Stratify mispredicted examples.

        TODO(Yoshi): Will avoid hand-crafted configuration

        Args:
            df (pd.DataFrame):

        Returns:
            es (ErrorSummary)

        """
        # Assume AutoConverter is mandatory for the function
        if self.ac is None:
            raise ValueError("AutoConverter instance is required to call",
                             "stratify_errors()")

        def calc_diversity(s):
            """Calculate entropy as a diversity metric."""
            probs = s / s.sum()
            return (probs * np.log(1.0 / probs)).sum()

        assert self.ac.hasdata
        error_df = self.get_mispredictions(df)

        # Conduct for loop for each column
        colname_list = []
        error_dist_df_list = []
        diversity_list = []
        sorted_colnames = sorted(error_df.columns.tolist())
        for colname in sorted_colnames:
            if colname not in self.ac.colname_type_dict:
                continue
            error_count_s = error_df[colname].value_counts()
            total_count_s = df[colname].value_counts()
            error_dist_df = pd.concat([error_count_s, total_count_s], axis=1)
            error_dist_df.columns = ["error_count", "total_count"]
            error_dist_df["error_rate"] = (error_dist_df["error_count"] /
                                           error_dist_df["total_count"])
            if len(error_dist_df) > max_numcat:
                continue

            error_dist_df.index.name = "group"
            error_dist_df = error_dist_df.reset_index()

            # Calculate diversity score
            diversity_score = calc_diversity(error_dist_df["error_rate"])
            error_dist_df.loc[:, 'colname'] = colname
            error_dist_df_list.append(error_dist_df)
            diversity_list.append(diversity_score)
            colname_list.append(colname)

        if len(error_dist_df_list) < 1:
            # No grouped result found
            # TODO(Yoshi): Output any message?
            return None

        error_dist_concat_df = pd.concat(error_dist_df_list, axis=0)
        error_dist_concat_df.set_index(["colname", "group"], inplace=True)
        diversity_df = pd.DataFrame({"diversity": diversity_list},
                                    index=colname_list)

        return ErrorSummary(error_dist=error_dist_concat_df,
                            diversity=diversity_df,
                            errors=error_df)

    def get_explanations(self,
                         test_df,
                         X=None,
                         topk=3,
                         max_candidates=10,
                         num_sampling=10,
                         spoil_method='random'):
        """Returns explanations (previously known as reason codes)

        V1 simply calculates the average difference of class probabilities
        no matter whether binary or multiclass

        Args:
            test_df (pd.DataFrame): Original DataFrame
            X (np.array): Test feature matrix
            topk (int): select top-k colnames for explanations
            max_candidates (int): At most <max_candidates> columns will be
                                  used for explanations (Default 10)
            num_sampling (int): Number of sampling iterations
                                (Default 10)
            spoil_method (str): {"random"}

        Returns:

        """
        # Assume AutoConverter is mandatory for the function
        if self.ac is None:
            raise ValueError("AutoConverter instance is required to call",
                             "get_explanations()")

        # TODO(Yoshi): spoil_method should be improved

        top_colnames = self.get_top_columns(n=max_candidates)

        # TODO(Yoshi): it's not straightforward to visualize representative
        #              values for subtables. Only focus on main table for now
        top_colnames = list(filter(lambda x: x.split('..')[0] == 'main',
                                   top_colnames))
        assert len(top_colnames) > 0

        table_colname_feature_pos_dict = self.get_feature_indexes()

        if X is None:
            assert self.ac.hasdata
            X = self.ac.X

        all_pred = self.alearn.predict_proba(X)

        table_colname_impact_dict = {}
        for table_colname in top_colnames:
            abs_diff_probs = np.zeros_like(all_pred)
            beg_idx, end_idx = table_colname_feature_pos_dict[table_colname]
            for _ in range(num_sampling):
                X_shuf = X.copy()
                np.random.shuffle(X_shuf[:, beg_idx:end_idx])
                all_pred_shuf = self.alearn.predict_proba(X_shuf)
                abs_diff_probs += np.abs(all_pred - all_pred_shuf)
            # <num_sample>-dimensional vector
            impact_scores = np.mean(abs_diff_probs, axis=1)
            table_colname_impact_dict[table_colname] = impact_scores
        impact_df = pd.DataFrame(table_colname_impact_dict)
        assert len(impact_df) == len(test_df)
        impact_df.index = test_df.index

        all_explanation_list = []
        for index, row in impact_df.iterrows():
            top_s = row.sort_values(ascending=False).head(topk)
            top_colnames = top_s.index.tolist()
            cur_explanation_list = []
            for table_colname in top_colnames:
                # split colanme in to tablename and colname
                tablename, colname = table_colname.split("..")
                val = test_df.ix[index][colname]
                cur_explanation_list.append((colname, val))
            all_explanation_list.append(cur_explanation_list)
        explain_df = pd.DataFrame({"explanations": all_explanation_list})
        assert len(explain_df) == len(test_df)
        explain_df.index = test_df.index
        return explain_df
