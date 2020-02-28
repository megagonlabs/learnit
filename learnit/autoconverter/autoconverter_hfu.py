import os
import sys
import unicodedata

from dateutil import parser
from keras.applications.densenet import DenseNet121
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
import numpy as np
import pandas as pd
import scipy as sp
from six import text_type
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import (FeatureUnion, _fit_one_transformer,
                              _fit_transform_one, _transform_one)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


if sys.version_info.major == 3:
    import pickle
else:
    import cPickle as pickle


class HeterogeneousFeatureUnion(FeatureUnion):

    def fit(self, X, y=None):
        """Fit all transformers using X.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data, used to fit transformers.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self : HeterogeneousFeatureUnion
            This estimator
        """
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_transformer)(trans, trans.select_item(X), y)
            for _, trans, _ in self._iter())
        self._update_transformer_list(transformers)
        return self

    def _validate_transformers(self):
        names, transformers = zip(*self.transformer_list)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t is None:
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform") or not hasattr(t, "select_item")):
                raise TypeError("All estimators should implement fit and "
                                "transform and select_item. "
                                "'%s' (type %s) doesn't" % (t, type(t)))

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, weight, trans.select_item(X), y,
                                        **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sp.sparse.issparse(f) for f in Xs):
            Xs = sp.sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, trans.select_item(X))
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sp.sparse.issparse(f) for f in Xs):
            Xs = sp.sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs


class ColumnTransformer(BaseEstimator, TransformerMixin):
    """Applies a pre-set transformer to a pre-set column

    """

    def __init__(self, colname):
        """Init

        Args:
            colname (str): column name of an input DataFrame

        """

        self.colname = colname
        self.tolist_flag = False

        if 'Vectorizer' in self.__class__.__name__:
            # e.g., TfidfVectorizer, CountVectorizer, TextLengthVectorizer
            self.tolist_flag = True

    def select_item(self, X):
        if self.tolist_flag:
            return X[self.colname].tolist()
        else:
            return X[self.colname].as_matrix()[None].T


class DummyTransformer(ColumnTransformer):
    """Dummy Transformer which returns original input."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_feature_names(self):
        return [u"Value"]


class CategoryOneHotEncoder(ColumnTransformer):
    """OneHotEncoder for String values

    Args:
    """

    def __init__(self, colname):
        super(CategoryOneHotEncoder, self).__init__(colname)
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder()

    def fit(self, X, y=None):
        X_encoded = self.label_encoder.fit_transform(X.ravel()).reshape(-1, 1)
        self.onehot_encoder.fit(X_encoded)
        return self

    def transform(self, X):
        X_encoded = self.label_encoder.transform(X.ravel()).reshape(-1, 1)
        return self.onehot_encoder.transform(X_encoded)

    def get_feature_names(self):
        # we need to transform classes_ to list of strings
        string_list = list(map(lambda x: text_type(x),
                           list(self.label_encoder.classes_)))
        return string_list


class DateTransformer(ColumnTransformer):
    """Transforms pandas datetime64 to feature vectors

    Args:
        weekday (boolean): weekday extraction flag. Default: True
        timeoftheday (boolean): specifies if we include hours and minutes.
        Default: True
        seconds (boolean): seconds extraction flag. Default: False
        microseconds (boolean): microseconds extraction flag. Default: False
        days_in_month (boolean): days in this month. Default: False
        is_leap_year (boolean): leap year indicator. Default: False
        month_start_end (boolean): month start and end indicators.
        Default: False
        nweek (boolean): ordinal of the week from the begginning of the year.
        Default: False

    TODO ideas:
        #idea: what we can do is apply all the possible parameters and filter
        out the ones that has equal values (say no milliseconds at all)
        holiday markers?
        season markers?
        time zones?

    """

    def __init__(self, colname,
                 weekday=True, timeoftheday=True, seconds=False,
                 microseconds=False, days_in_month=False,
                 is_leap_year=False, month_start_end=False, nweek=False):
        super(DateTransformer, self).__init__(colname)

        self.weekday = weekday
        self.timeoftheday = timeoftheday
        self.seconds = seconds
        self.microseconds = microseconds
        self.days_in_month = days_in_month
        self.is_leap_year = is_leap_year
        self.month_start_end = month_start_end
        self.nweek = nweek

    def fit(self, X, y=None):
        """Fit function

        Args:
            X (pandas dataframe): column to transform

        Returns:
            self

        """

        return self

    def transform(self, X):
        """Transform function

        Args:
            X (pandas.DataFrame): df to pick the self.colname column

        """

        dates = pd.DatetimeIndex(X[self.colname])
        result_df = pd.DataFrame(dates.year.values, columns=['Year'])
        result_df['Month'] = dates.month.values
        result_df['Day'] = dates.day.values
        result_df['DayOfYear'] = dates.dayofyear.values  # ordinal day of year

        if self.weekday:
            result_df['Weekday'] = dates.dayofweek.values

        if self.days_in_month:
            result_df['Days_in_month'] = dates.days_in_month.values

        if self.is_leap_year:
            # *1 is converting bool to int
            result_df['is_leap_year'] = dates.is_leap_year * 1

        if self.month_start_end:
            # *1 is converting bool to int
            result_df['Month_start'] = dates.is_month_start * 1
            result_df['Month_end'] = dates.is_month_end * 1

        if self.nweek:
            result_df['Nweek'] = dates.week.values  # ordinal week of the year

        if self.timeoftheday:
            result_df['Hour'] = dates.hour.values
            result_df['Minutes'] = dates.minute.values
            if self.seconds:
                result_df['Seconds'] = dates.second.values
            if self.microseconds:
                result_df['Microseconds'] = dates.microsecond.values

        # unix date
        result_df['Unix'] = dates.astype(np.int64)

        return result_df.as_matrix()

    def get_feature_names(self):
        """Provides column names for features."""

        # The following columns initialization is needed to make
        # get_feature_names function work properly before fit/transform
        self.columns = ['Year', 'Month', 'Day', 'DayOfYear']
        if self.weekday:
            self.columns.append('Weekday')

        if self.days_in_month:
            self.columns.append('Days_in_month')

        if self.is_leap_year:
            self.columns.append('is_leap_year')

        if self.month_start_end:
            self.columns.append('Month_start')
            self.columns.append('Month_end')

        if self.nweek:
            self.columns.append('Nweek')

        if self.timeoftheday:
            self.columns.append('Hour')
            self.columns.append('Minutes')
            if self.seconds:
                self.columns.append('Seconds')
            if self.microseconds:
                self.columns.append('Microseconds')

        self.columns.append('Unix')

        return [text_type(x) for x in self.columns]

    def select_item(self, X):
        # TODO(Bublin): pass pd.Series instead of pd.DataFrame?
        return X


class SklearnVectorizer(ColumnTransformer):
    """Converts into topic distributions using LDA."""

    def __init__(self, colname, vectorizer=None, **kwargs):
        """Initializer

        Args:
            kwargs_dict (dict): kwargs parameters to CountVectorizer and
                                LatentDirichletAllocation

        """
        super(SklearnVectorizer, self).__init__(colname)
        self.vectorizer = vectorizer(**kwargs)

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


class LDAVectorizer(ColumnTransformer):
    """Converts into topic distributions using LDA."""

    def __init__(self, colname, kwargs_dict=None):
        """Initializer

        Args:
            kwargs_dict (dict): kwargs parameters to CountVectorizer and
                                LatentDirichletAllocation

        """
        super(LDAVectorizer, self).__init__(colname)
        if kwargs_dict is None:
            kwargs_dict = {'CountVectorizer': {},
                           'LatentDirichletAllocation': {}}

        self.vectorizer = CountVectorizer(
            **kwargs_dict['CountVectorizer'])
        self.lda = LatentDirichletAllocation(
            **kwargs_dict['LatentDirichletAllocation'])

    def fit(self, X, y=None):
        assert type(X) == list
        self.vectorizer.fit(X)
        self.lda.fit(self.vectorizer.transform(X))
        return self

    def transform(self, X):
        assert type(X) == list
        return self.lda.transform(self.vectorizer.transform(X))

    def get_feature_names(self):
        # To make sure self.lda is fitted (is there any better way?)
        assert self.lda._n_components is not None
        clsname = self.__class__.__name__
        return list(map(lambda x: u"{}_{}".format(clsname, x),
                        range(self.lda.n_components)))


class TextLengthVectorizer(ColumnTransformer):
    """Extracts textual length.

    Future work: Word length vs character length

    """

    def __init__(self, colname, null_value=0):
        """Initialization

        Args:
            null_value (int): Default value for empty string

        """
        super(TextLengthVectorizer, self).__init__(colname)
        self.null_value = null_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transform function

        Args:
            X (list): list of string objects

        Returns:
            np.array

        """
        assert type(X) == list
        return np.array(list(map(lambda x: len(x), X))).reshape(-1, 1)

    def get_feature_names(self):
        return [u"text_length"]


class PreTrainedModel:
    """Singleton pre-trained model

    All models will be loaded at first calling

    """

    model_dict = {}

    def __getitem__(self, key):
        if key not in self.model_dict:
            method_name = "_add_{}".format(key)
            getattr(self, method_name)()

        return self.model_dict[key]

    @classmethod
    def _add_resnet50(cls):
        cls.model_dict["resnet50"] = {
            "name": "resnet50",
            "model": ResNet50(weights='imagenet', include_top=False),
            "image_size": (224, 224)
        }

    @classmethod
    def _add_densenet121(cls):
        cls.model_dict["densenet121"] = {
            "name": "densenet121",
            "model": DenseNet121(weights='imagenet', include_top=False),
            "image_size": (224, 224)
        }

    @classmethod
    def _add_mobilenet(cls):
        cls.model_dict["mobilenet"] = {
            "name": "mobilenet",
            "model": MobileNet(
                weights='imagenet', include_top=False,
                input_shape=(224, 224, 3), pooling="avg"
            ),
            "image_size": (224, 224)
        }

    @staticmethod
    def get_output_dimension(model):
        len_output = len(model.output.shape)
        dim = model.output.shape[len_output - 1]

        return dim


class ImageVectorizer(ColumnTransformer):
    """Extracts image vector with predefined Keras DL Model.

    Default model: image net
    Input: url list
    Output: The last layer vector

    """

    def __init__(self, colname, way="mobilenet", batch_size=512):
        super(ImageVectorizer, self).__init__(colname)
        self.trained_model = PreTrainedModel()[way]
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transform function

        Args:
            X (np.array): list of string objects

        Returns:
            np.array

        """
        X = np.array(X)
        loop_size = X.shape[0] // self.batch_size
        loop_size += ((X.shape[0] % self.batch_size) > 0)

        return np.array([self.img2vec(x) for x in np.split(X, loop_size)])

    def img2vec(self, path_list):
        """Transform function

        Args:
            path_list (np.array): path url list

        Returns:
            np.array

        """
        img_list = []
        for path in path_list:
            if not path.startswith("/"):
                path = os.getcwd() + "/" + path
            img = image.load_img(path,
                                 target_size=self.trained_model["image_size"])
            x = image.img_to_array(img)
            x = preprocess_input(x)
            img_list.append(x)
        img_arr = np.array(img_list)
        pred = self.trained_model["model"].predict(img_arr)
        for i in range(len(self.trained_model["model"].output.shape) - 1):
            pred = pred[0]

        return pred

    def get_feature_names(self):
        # To make sure self.lda is fitted (is there any better way?)
        clsname = self.__class__.__name__
        dim = PreTrainedModel.get_output_dimension(self.trained_model["model"])
        return [u"{}_{}_{}".format(clsname, self.trained_model["name"], x)
                for x in range(dim)]


class AutoConverter():
    def __init__(self,
                 target,
                 strategy='auto',
                 coltype_converters={},
                 column_converters={},
                 use_column_converter_only=True,
                 n_jobs=1):
        """Big wrapping class for convertors

        Args:
            target (str): target column name
            strategy (str): {'auto'}
            coltype_converters (dict): dict of customized Transformers
            column_converters (dict): dict of customized column transformers
            use_column_converter_only (bool): Use only column converter or not
            n_jobs (int): n_jobs parameter for FeatureUnion

        column_converters will be applied to columns on a priority basis.
        If use_column_converter == True (default value),
        pre-defined transformers in TransformerCatalog will NOT be applied.

        Therefore, giving an empty list to a column can be used to "ignore"
        the column for feature extraction.

        In the following example, only TfIdfVectorizer with default parameters
        will be applied to "Name" column and no transformer will be applied to
        "Age" column.

        column_converters={"Name": [(TfIdfVectorizer, {})],
                           "Age": []}

        """

        self.target = target
        self.strategy = strategy
        self.feature_names = []
        self.X = None
        self.y = None
        self.hasdata = False
        self.target_le = LabelEncoder()
        self.subtables_ = None
        self.converter_catalog = None
        self.set_converter(coltype_converters)
        self.column_converters = column_converters
        self.use_column_converter_only = use_column_converter_only
        self.n_jobs = n_jobs

    def set_converter(self, coltype_converters):
        """Insert customized transformers into self.converter_catalog."""
        # TODO(Yoshi): Technically, dict.update overwrite existing entry
        # We might want to "append" instead. To be discussed.
        self.converter_catalog = (DefaultTransformerCatalog.transformer_dict
                                  .copy())
        self.converter_catalog.update(coltype_converters)

    def fit(self, df, subtables=None, y=None, custom_types={}):
        """Fits the data to the custom_converters

        Args:
            df (pd.DataFrame): main dataframe table

            subtables (dictionary): dictionary of subtables with keys for
                linking them to main table. Default: None.
                subtables =
                    {tabname(str) : { "table": (pd.Dataframe),
                        "link_key": (str) main table column name,
                        "group_key": (str) this table column name,
                        "custom_aggregator": (dict) col_type:aggregator_class}}
                Example:
                    {"school_table": {"table": school_df,
                                      "link_key": "school_id",
                                      "group_key": "id",
                                      "custom_aggregator": {"text":
                                          CustomTextAggregator()}
                                     }
                    }
            custom_types (dictionary): dictionary of col_types that overrides
                col_type_dicts made by auto_converter orcibly

        Returns:
            self

        """
        assert self.target in df

        # filtering None
        df.dropna(subset=[self.target], inplace=True)
        # filterung NaN
        df = df[df[self.target].notnull()]

        self.target_le.fit(df[self.target].as_matrix())

        X_df = df.drop(self.target, axis=1)

        # 1. typing columns
        self.colname_type_dict = type_columns(X_df)
        if isinstance(custom_types, dict):
            self.colname_type_dict.update(custom_types)

        # 2. Pre-imputing missing values for textual column
        for colname in X_df.columns:
            if (self.colname_type_dict[colname] == 'text'
                    or self.colname_type_dict[colname] == 'categorical'
                    or self.colname_type_dict[colname] == 'text_ja'):
                X_df.loc[:, colname] = X_df[colname].fillna("NaN").astype(str)

        # 3. create feature union
        transformer_list = []
        for colname in X_df.columns:

            if colname in self.column_converters:
                for transformer_cls, kwargs in self.column_converters[colname]:
                    transformer_list.append(
                        (u"{}.{}".format(colname, transformer_cls.__name__),
                         transformer_cls(colname, **kwargs))
                    )
                if self.use_column_converter_only:
                    # Since transformer(s) are defined by users,
                    # skip automatic assignment of transformers for this column
                    continue

            assert colname in self.colname_type_dict
            coltype = self.colname_type_dict[colname]

            if coltype == 'ignore':
                continue

            if coltype == 'date':
                # we don't want to pass np array to date transformer,
                # instead we pass pandas df
                # TODO(Yoshi): This is hard-coded??
                d = DateTransformer(colname=colname)
                transformer_list.append((u"{}.{}".format(colname, 'date'), d))
                continue

            t_dict = self.converter_catalog[coltype]
            for transformer in t_dict:
                transformer_cls = transformer[0]
                kwargs = transformer[1]
                transformer_list.append(
                    (u"{}.{}".format(colname, transformer_cls.__name__),
                     transformer_cls(colname, **kwargs))
                )

        # 4. fit feature union
        if transformer_list:  # if there's something to transform
            self.fu = HeterogeneousFeatureUnion(transformer_list,
                                                n_jobs=self.n_jobs)
            self.fu.fit(X_df)

            feature_names = list(map(lambda x: 'main..' + text_type(x),
                                 self.fu.get_feature_names()))
        else:  # emppty main table (only target and ignore types)
            # we assume there exist information in subtables then
            if not subtables:
                raise ValueError("There's nothing to transform")
            self.fu = None
            feature_names = []

        # defining Aggregator structure and fitting the tables in
        if subtables:
            self.subtables_ = subtables

            for key in sorted(list(subtables.keys())):
                subtable_dict = subtables[key]

                if subtable_dict['link_key'] not in X_df.columns:
                    raise KeyError("Link key " + subtable_dict['link_key'] +
                                   " does not exist in the main table")

                aggr = AutoAggregator(
                    group_key=subtable_dict['group_key'],
                    custom_aggregators=subtables.get("custom_aggregator", {}))
                self.subtables_[key]['aggr'] = aggr
                aggr.fit(subtable_dict['table'])
                self.colname_type_dict[key] = aggr.colname_type_dict.copy()

                # gathering feature names from subtables
                append_list = list(
                    map(lambda x: text_type(key) + '..' + text_type(x),
                        aggr.feature_names))
                feature_names.extend(append_list)

        self.feature_names = feature_names

        return self

    def transform(self,
                  df,
                  subtables=None,
                  prediction=False):
        """Transforms data to feature matrix

        Args:
            df (pandas.DataFrame): data to transform
            subtables (dictionary): dictionary of subtables with keys for
                linking them to main table. Default: None.
                subtables =
                    {tabname(str) : { "table": (pd.Dataframe),
                        "link_key": (str) main table column name,
                        "group_key": (str) this table column name }}
                Example:
                    {"school_table": {"table": shool_pd},
                                      "link_key": "school_id",
                                      "group_key": "id" }
                    }

            prediction (bool): Returns only X if True

        Returns:
            X (numpy.ndarray): feature matrix
            y (array-like of shape [n_samples]): target vector

        """

        if not prediction:
            # filtering None
            df.dropna(subset=[self.target], inplace=True)
            # filterung NaN
            df = df[df[self.target].notnull()]

            # TODO(Yoshi): Should display Warning message when transform
            # is called with prediction=False if self.hasdata is True
            if self.hasdata:
                print("[WARNING] This instance already has been fitted.")
            assert self.target in df
            y_unique = df[self.target].unique()

            if len(y_unique) == 1 and np.isnan(y_unique[0]):
                # this just leaves y equal to a np.nan vector of the same size
                # TODO(Yoshi): This should raise exception.
                #              Will revise here after specifying exceptions
                y = df[self.target]
            else:
                y = self.target_le.transform(df[self.target].as_matrix())

            X_df = df.drop(self.target, axis=1)

        else:
            # Prediction
            if self.subtables_ is not None:
                assert subtables is not None

            if self.target in df:
                X_df = df.drop(self.target, axis=1)
            else:
                X_df = df

        # TODO(later): Pre-imputing. This part could be redundant
        for colname in X_df.columns:
            if self.colname_type_dict[colname] in ['categorical',
                                                   'text',
                                                   'text_ja']:
                X_df.loc[:, colname] = X_df[colname].fillna("NaN").astype(str)

        if self.fu:
            X = self.fu.transform(X_df)
        else:
            # Creating the empty matrix of the same size to use it later during
            # data aggregation, since we can't use feature union in absence of
            # features
            X = np.empty([X_df.shape[0], 0])

        # Ad-hoc way to convert sparse matrix into numpy.array and replace NaN
        # values with 0.0
        if type(X) == sp.sparse.csr.csr_matrix:
            X = X.toarray()
        X[np.isnan(X)] = 0.0

        # transforming subtables and concating them with main table feature
        # matrix
        if subtables:
            # TODO(Kate): make sure that subtables passed and subtables stored
            # have the same structure. Any ideas?
            X_gather = pd.DataFrame(X)

            for key in sorted(list(subtables.keys())):
                subtable = subtables[key]
                aggr = subtable['aggr']
                link_key = subtable['link_key']
                X_sub = aggr.transform(subtable['table'])
                # combine X_gather with subtable['link_key']

                if link_key in X_gather.columns.tolist():
                    raise KeyError(
                        'column already exists in a dataframe' + link_key)

                X_gather[link_key] = df[link_key]
                # X_sub is already a pd.DataFrame with group_key included
                # as index
                X_gather = X_gather.merge(X_sub, how='left', left_on=link_key,
                                          right_index=True)
                # make sure we don't leave anything(index) behind ;)
                del X_gather[link_key]

                # do something with get_feature_names

            X = X_gather.as_matrix()

        # TODO(Yoshi): Post pre-processing such as missing value imputation
        # TODO(Yoshi): Tentative naive replacement of NaN values
        X = np.nan_to_num(X)

        if not prediction:
            self.X = X
            self.y = y
            self.hasdata = True
            return [X, y]
        else:
            return X

    def fit_transform(self, df, subtables=None, y=None):
        """Fit + Transform

        Args:
            df (pandas.DataFrame): main df
            subtables (dict): dictionary of subtables

        Returns:
            X (numpy.ndarray): feature matrix
            y (array-like of shape [n_samples]): target vector

        """

        return self.fit(df, subtables).transform(df, subtables)

    def index2label(self, predictions):
        """Transforms predictions from numerical format back to labels

        Args:
            predictions (np.array): array of label numbers

        Returns:
            labels (np.array): array of label values

        """

        return self.target_le.inverse_transform(predictions)

    def get_feature_names(self, colname=None):
        """Returns feature names

        Args:
            colname (str or tuple): column name
            if colname is a tuple (subtable name, colname)
            if None returns all feature names
            (default: None)

        Returns:
            feature_names (list)

        """

        if colname is None:
            if len(self.feature_names) == 0:
                # TODO(Yoshi): Do we want to use a "trained" flag instead?
                print("[WARNING]:",
                      "AutoConverter instance has extracted no feature.",
                      "Probably, it has not been fit to data yet.")
            return self.feature_names

        # Use tuple (or list) to handle subtable feature names
        if type(colname) in [tuple, list]:
            # TODO(Yoshi): replace with Exception
            assert len(colname) == 2
            colname_ = "..".join(colname)
        else:
            # colname is in main table
            colname_ = "main..{}".format(colname)

        colname_idx_list = list(filter(lambda x: colname_ in x[1],
                                       enumerate(self.feature_names)))
        colname_list = list(map(lambda x: x[1], colname_idx_list))

        return colname_list

    def save(self,
             filepath,
             overwrite=False):
        """Save AutoConverter object as pickle file

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
    def load(cls, filepath):
        """Load AutoConverter object from pickle file

        Args:
            filepath (str): Input pickle filepath

        Returns:
            AutoLearn object

        """
        with open(filepath, "rb") as fin:
            obj = pickle.load(fin)
        assert obj.__class__.__name__ == 'AutoConverter'
        return obj


class DefaultTransformerCatalog:
    # TODO(later): boolean
    transformer_dict = {
        'text': [
            (SklearnVectorizer, {"vectorizer": TfidfVectorizer,
                                 "max_features": 500}),
            (LDAVectorizer, {}),
            (TextLengthVectorizer, {})
        ],
        'text_ja': [
            (SklearnVectorizer, {"vectorizer": TfidfVectorizer,
                                 "max_features": 500,
                                 "analyzer": "char_wb",
                                 "ngram_range": (2, 3)}),
            (TextLengthVectorizer, {})
        ],
        'numerical': [(DummyTransformer, {})],
        'categorical': [(CategoryOneHotEncoder, {})],
        'date': [(DateTransformer, {})]
    }


class AutoAggregator():
    def __init__(self,
                 group_key,
                 custom_aggregators={},
                 n_jobs=1):
        """Initialization

        Args:
            group_key (string): column name of the aggregated table. This
                column will be used for 'grouping'/aggregating the values in
                the dataframe
            custom_aggregators (list): list of customized Aggregators
            n_jobs (int): n_jobs parameter for FeatureUnion

        Class performs feature extraction from the subsidiary table and then
        groups the rows of the data by group_key

        """
        self.group_key = group_key
        self.feature_names = []
        self.set_aggregator_catalog(custom_aggregators)
        self.n_jobs = n_jobs

    def set_aggregator_catalog(self, custom_aggregators):
        """Insert customized aggregators to self.aggregator_catalog."""
        # TODO(Yoshi): Technically, dict.update overwrite existing entry
        # We might want to "append" instead. To be discussed.
        self.aggregator_catalog = (DefaultAggregatorCatalog.transformer_dict
                                   .copy())
        self.aggregator_catalog.update(custom_aggregators)

    def fit(self, sec_df):
        """Fits the data

        Args:
            sec_df (pandas.DataFrame): subsidiary table

        """

        # TODO(Kate): make sure that df has group_key as index so that when we
        # pass the whole thing to transformers, we could group by it

        # 1. typing columns
        self.colname_type_dict = type_columns(sec_df)

        # 2. Pre-imputing missing values for textual column
        for colname in sec_df.columns:
            if (self.colname_type_dict[colname] == 'text'
                    or self.colname_type_dict[colname] == 'categorical'
                    or self.colname_type_dict[colname] == 'text_ja'):
                sec_df.loc[:, colname] = sec_df[colname].fillna("NaN")

        # 3. create feature union
        transformer_list = []
        for colname in sec_df.columns:
            assert colname in self.colname_type_dict
            coltype = self.colname_type_dict[colname]

            if colname == self.group_key:
                # we are ignoring this column, since it should be included
                # from the main table
                continue

            if coltype == 'ignore':
                continue

            if coltype in self.aggregator_catalog:
                for agg_cls, kwargs in self.aggregator_catalog[coltype]:
                    agg = agg_cls(colname=colname,
                                  group_key=self.group_key,
                                  **kwargs)
                    transformer_list.append(
                        (u"{}.{}".format(colname, coltype), agg)
                    )

        # 4. fit feature union
        # we need to make sure that we still store id values from this one
        self.fu = HeterogeneousFeatureUnion(transformer_list,
                                            n_jobs=self.n_jobs)
        self.fu.fit(sec_df)
        self.feature_names = self.fu.get_feature_names()

        return self

    def transform(self, sec_df):
        """Transformer

        Args:
            df (pandas.DataFrame): secondary dataframe for transformation and
            aggregation
        """

        X = self.fu.transform(sec_df)
        # change X back to pandas.DataFrame
        X_df = pd.DataFrame(X)
        X_df.index = sorted(sec_df[self.group_key].unique())

        return X_df

    def fit_transform(self, df, y=None):
        return self.fit(df).transform(df)

# I copied this from AutoConvertor. We may want to move it out of both classes


def is_japanese(string):
    """Returns whether the text include Japanese

        Args:
            string (str): value

        Returns:
            is_ja (bool): whether the text include Japanese or not

    """
    if not isinstance(string, text_type):
        return False

    if hasattr(string, 'decode'):
        # In Python 2.x, string has to be decoded as Unicode
        string = string.decode("utf-8")

    for ch in string:
        try:
            name = unicodedata.name(ch)
            for symbolname in ["CJK UNIFIED", "HIRAGANA", "KATAKANA"]:
                if symbolname in name:
                    return True
        except Exception as e:
            # The character is not defined in UTF-8
            # TODO(Yoshi): Logger
            print(e)

    return False


def is_japanese_col(sr, check_num=500):
    """Returns whether the text include Japanese

    Args:
        sr (pd.Series): text Series
        check_num(int): number of check rows from

    Returns:
        is_ja_column (bool): whether the series is Japanese or not

    """
    use_sr = sr[:check_num]
    for index, string in use_sr.iteritems():
        if is_japanese(string):

            return True

    return False


def type_column(s, cat_proportion_threshold=0.01,
                cat_absolute_threshold=100, id_threshold=0.99):
    """Returns type of column

    Args:
        s (pd.Series): column values

        cat_proportion_threshold (float): percentage of unique values in
            dataset. Default 0.01
        cat_absolute_threshold (int): absolute value count. Default = 100.
            The minimum of cat_proportion_threshhold and
            cat_absolute_threshold is used to define the actual threshhold.
        id_threshold (float): threshold to ignore id columns.
            Default = 0.99

    Returns:
        coltype (str): type of the column. Current possible values:
            numerical','categorical', 'text', 'date', 'ignore'. The latter
            column type is used to ignore the column when extracting features.
            Currently used for id-containing columns.

    """
    coltype = None
    if s.dtype in ['float16', 'float32', 'float64']:
        coltype = 'numerical'
    elif s.dtype == 'datetime64[ns]':
        coltype = 'date'
    elif (s.value_counts().size - 1 <
            min(cat_proportion_threshold * s.size, cat_absolute_threshold)):
        # valid both for integer or text values
        coltype = 'categorical'
    elif s.dtype in ['intc', 'intp', 'int8', 'int16', 'int32', 'int64',
                     'uint8', 'uint16', 'uint32', 'uint64']:
        if s.value_counts().size - 1 > id_threshold * s.size:
            # this is probably id and should be ignored
            coltype = 'ignore'
        else:
            coltype = 'numerical'
    else:
        # object type
        try:
            # Another chance to determine if "date" type
            # TODO(Yoshi): Will make the sampling size a parameter
            # parser.parse() raises Exception if it fails to parse
            s[:100].apply(lambda x: parser.parse(x))
            coltype = 'date'
        except Exception:
            None

        if coltype is None:
            if s.str.count(' ').sum() == 0:
                # seems like id
                coltype = 'ignore'
            else:
                if is_japanese_col(s):
                    coltype = 'text_ja'
                else:
                    coltype = 'text'

    return coltype


def type_columns(df):
    """Returns column names for the feature matrix

    Args:
        df (pd.DataFrame): input data

    Returns:
        colname_type_dict (dict)

    """

    # we need to make sure we add aggr columns here ----------------------
    colname_type_dict = {}
    for c in df.columns:
        colname_type_dict[c] = type_column(df[c])
    return colname_type_dict


class AggregateTransformer(BaseEstimator, TransformerMixin):
    """Applies a pre-set transformer to a pre-set column

    """

    def __init__(self, colname, group_key):
        """Init

        Args:
            colname (str): column name of an input DataFrame

        """

        self.colname = colname
        self.group_key = group_key

    def select_item(self, X):

        return X[[self.group_key, self.colname]]


class NumericalAggregator(AggregateTransformer):
    """Transforms numerical values to basic statistics.

    """

    def __init__(self, colname, group_key, functions=None):
        """Initialization

        Args:
            colname (str): Column name used for aggregation
            group_key (str):  Column name used for groupby
            functions (dict): Dictionary of function names and functions
              Default: {'sum': np.sum,
                        'mean': np.mean,
                        'std': np.std,
                        'count': 'count'}

        """
        super(NumericalAggregator, self).__init__(colname, group_key)

        if functions is None:
            functions = {'sum': np.sum,
                         'mean': np.mean,
                         'std': np.std,
                         'count': 'count'}
        self.functions = functions
        assert colname != group_key, "group key  equal to colname"

    def fit(self, X, y=None):
        """Fit function

        Args:
            X (pandas dataframe): df to pick colname from

        Returns:
            self

        """

        return self

    def transform(self, X):
        """Transform function

        Args:
            X (pandas.DataFrame): df to pick colname from

        Returns:
            result (pandas.DataFrame): aggregated values
        """

        grouped = X.groupby(self.group_key)
        result_df = grouped.agg(self.functions.values())

        # Drop top-level columns to sort by function names
        result_df.columns = result_df.columns.droplevel()
        result_df = result_df[sorted(self.functions.keys())]

        # Make sure returned columns are consistent with key names of functions
        assert sorted(self.functions.keys()) == result_df.columns.tolist()

        return result_df

    def get_feature_names(self):
        """Provides column names for features."""

        return[text_type(x) for x in sorted(self.functions.keys())]


class TextualAggregator(AggregateTransformer):
    """Aggregates textual values."""
    def __init__(self, colname, group_key, vectorizer=None):
        """Initialization

        Args:
            colname (str): Column name used for aggregation
            group_key (str):  Column name used for groupby
            vectorizer (Vectorizer):
                sklearn.feature_extraction.text.
                  CountVectorizer()
        """
        super(TextualAggregator, self).__init__(colname, group_key)

        assert colname != group_key
        if vectorizer is None:
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer = vectorizer

    def fit(self, df, y=None):
        """Fit function

        Args:
            df (pd.DataFrame): df to pick colname from

        Returns:
            self
        """

        assert self.colname in df.columns
        assert self.group_key in df.columns

        # Concatenate textual contents of a user into a single string
        texts = df.groupby([self.group_key]).agg(
            {self.colname: " ".join})[self.colname].tolist()
        self.vectorizer.fit(texts)

        return self

    def transform(self, df):
        """Transform function

        Args:
            df (pd.DataFrame): df to pick colname from

        Returns:
            result_df (pd.DataFrame): Extracted features

        """
        assert self.colname in df.columns
        assert self.group_key in df.columns

        # Same above
        agg_df = df.groupby([self.group_key]).agg({self.colname: " ".join})
        texts = agg_df[self.colname].tolist()

        X = self.vectorizer.transform(texts).todense()
        result_df = pd.DataFrame(X)
        result_df.index = agg_df.index

        # The original order of the DataFrame should be preserved.
        return result_df

    def get_feature_names(self):
        """Returns feature names."""
        return self.vectorizer.get_feature_names()


class CategoryAggregator(AggregateTransformer):
    """Transforms a column of categorical values to a row of tfidf values."""

    def __init__(self, colname, group_key, vectorizer=None):
        super(CategoryAggregator, self).__init__(colname, group_key)
        assert colname != group_key, "group key  equal to colname"

        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        else:
            self.vectorizer = vectorizer

    def fit(self, X, y=None):
        """Fit function

        Args:
            X (pandas dataframe): df to pick colname from

        Returns:
            self

        """

        df = X[[self.group_key, self.colname]]
        cat_values = list(df[self.colname])
        # cat values might be numbers, so we force them to strings
        cat_values = list(map(lambda x: text_type(x), cat_values))
        text = ' '.join(cat_values)
        self.vectorizer.fit([text])

        return self

    def transform(self, X):
        """Transform function

        Args:
            X (pandas.DataFrame): df to pick colname from

        Returns:
            result (pandas.DataFrame): aggregated values
        """

        df = X[[self.group_key, self.colname]]
        grouped = df.groupby(self.group_key)
        grouped_agg = grouped.agg(" ".join)
        result = self.vectorizer.transform(grouped_agg[self.colname])
        result_df = pd.DataFrame(result.todense())

        return result_df

    def get_feature_names(self):
        """Provides column names for features."""

        return self.vectorizer.get_feature_names()


class DefaultAggregatorCatalog:
    ja_vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    transformer_dict = {'text': [(TextualAggregator, {}),
                                 (LDAVectorizer, {})],
                        'text_ja': [(TextualAggregator,
                                     {"vectorizer": ja_vec})],
                        'numerical': [(NumericalAggregator, {})],
                        'categorical': [(CategoryAggregator, {})]}


def check_transformer(input_df, colname, transformer):
    """Check if a user-defined tarnsformer works properly

    Args:
        input_df (pd.DataFrame):
        colname (str): target column name
        transformer (Transformer): Transformer instance

    Returns:

    """
    # TODO(Yoshi): Replace with exception
    assert colname in input_df

    try:
        X = transformer.fit_transform(transformer.select_item(input_df))
    except Exception as e:
        print("Something wrong happened. :(")
        print(e)
        return None

    print("Feature(s) successfully extracted! :)")
    print("# of features={}".format(X.shape[1]))
    return X
