# -*- coding: utf-8 -*-
import datetime
import os
import shutil
import unittest

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from learnit.autoconverter.autoconverter import AutoConverter
from learnit.autoconverter.autoconverter import CategoryAggregator
from learnit.autoconverter.autoconverter import CategoryOneHotEncoder
from learnit.autoconverter.autoconverter import DateTransformer
from learnit.autoconverter.autoconverter import LDAVectorizer
from learnit.autoconverter.autoconverter import TextLengthVectorizer
from learnit.autoconverter.autoconverter import TextualAggregator
from learnit.autoconverter.autoconverter import type_column
# unittest.TestCase.setUp()


class DummyTransformerTestCase(unittest.TestCase):
    # def setUp(self):
    # super(TestFunctions, self).setUp()

    def test_fit(self):
        pass
        # def test_fit_dimension(...)

    def test_transform(self):
        pass
        # X, y = get_dataset('boston')
        # clf = LogisticRegression()
        # result_info = run_cross_validation(X, y, clf=clf)

    # def tearDown(self):
    #    super(TestFunctions, self).tearDown()


class DateTransformerTestCase(unittest.TestCase):
    def setUp(self):
        super(DateTransformerTestCase, self).setUp()

        base = datetime.datetime(2016, 1, 1)
        arr = np.array([
            base + datetime.timedelta(days=350 * i, hours=i, minutes=2 * i,
                                      seconds=3 * i, microseconds=4 * i)
            for i in range(5)
        ])
        self.df = pd.DataFrame(arr, columns=['Date'])
        self.df['SomethingElse'] = np.array([(10 + i) for i in range(5)])

    # def test_fit(self):

    # def test_fit_dimension(...)

    def test_transform(self):

        dt = DateTransformer(colname='Date')
        result = dt.transform(self.df)

        self.assertEqual(result.shape, (5, 8), 'Incorrect array shape')
        self.assertEqual(result.sum(), 7560469230000051530,
                         'Date transformation unsuccessful')

        dt1 = DateTransformer(colname='Date',
                              weekday=False, timeoftheday=False)
        result1 = dt1.transform(self.df)
        self.assertEqual(result1.shape, (5, 5),
                         'Weekday or timeoftheday flags not working properly')

        dt2 = DateTransformer(colname='Date', days_in_month=True,
                              is_leap_year=True, month_start_end=True,
                              nweek=True, seconds=True)
        result2 = dt2.transform(self.df)
        self.assertEqual(result2.shape, (5, 14))
        self.assertEqual(result2.sum(), 7560469230000051959,
                         'Date Transformation wrong')

    def test_get_feature_names(self):
        dt = DateTransformer(colname='Date')
        dt.transform(self.df)

        self.assertEqual(dt.get_feature_names(),
                         ['Year', 'Month', 'Day', 'DayOfYear', 'Weekday',
                          'Hour', 'Minutes', 'Unix'])

    # def tearDown(self):
    # super(TestFunctions, self).tearDown()


class LDAVectorizerTestCase(unittest.TestCase):
    def setUp(self):
        self.vectorizer = LDAVectorizer()
        self.assertIsNotNone(self.vectorizer)

    def test_fit_transform(self):
        texts = ['I have a pen.', 'I have an apple.', 'Uh! apple-pen.']
        X = self.vectorizer.fit_transform(texts)
        self.assertTrue(X.shape, (3, 10))


class TextLengthVectorizerTestCase(unittest.TestCase):
    def setUp(self):
        self.vectorizer = TextLengthVectorizer()
        self.assertIsNotNone(self.vectorizer)

    def test_fit_transform(self):
        texts = ['I have a pen.', 'I have an apple.', 'Uh! apple-pen.']
        X = self.vectorizer.fit_transform(texts)
        self.assertEqual(X.shape, (3, 1))
        self.assertEqual(self.vectorizer.get_feature_names(),
                         ["text_length"])
        self.assertTrue(np.array_equal(X,
                                       [[13], [16], [14]]))


class CategoryOneHotEncoderTestCase(unittest.TestCase):
    def setUp(self):
        self.X = np.array(['A', 'B', 'A', 'B', 'C'])
        # adding unseen '3'
        self.X1 = np.array(['A', 'C', 'Unicorn'])

    def test_transform(self):

        encoder = CategoryOneHotEncoder(ignore=True)
        encoder.fit(self.X)
        self.assertEqual(len(encoder.get_feature_names()), 3,
                         "incorrect features of encoder")

        feature_columns = encoder.transform(self.X).todense()

        self.assertEqual(feature_columns.shape, (5, 3),
                         "incorrect dataframe shape")
        new_feature_columns = encoder.transform(self.X1)
        self.assertEqual(new_feature_columns.shape, (3, 3),
                         "incorrect dataframe shape")

        encoder1 = CategoryOneHotEncoder(ignore=False)
        encoder1.fit(self.X)
        self.assertEqual(len(encoder1.get_feature_names()), 4,
                         "incorrect features of encoder")

        new_feature_columns = encoder1.transform(self.X1).todense()
        self.assertEqual(new_feature_columns.shape, (3, 4),
                         "incorrect dataframe shape")


class AutoConverterTestCase(unittest.TestCase):
    def setUp(self):
        # add testing data here
        df_main = pd.DataFrame({'A': np.array([1, 2, 3, 4, 5]),
                                'B': pd.Timestamp('20130102'),
                                'Target': np.array([1, 0, 1, 0, np.nan]),
                                'E': pd.Categorical(
                                    ["test", "train", "test", "train", "test"])
                                })

        df_sub1 = pd.DataFrame({'X': np.array([4, 2, 2, 1, 5]),
                                'Y': np.array([10, 20, 30, 40, 59],
                                              dtype='float64'),
                                'Z': pd.Categorical(
                                    ["fish", "chicken", "chicken",
                                     "fish", "milk"])})
        df_sub2 = pd.DataFrame({'X': pd.Categorical(['train', 'test', 'test']),
                                'Y': np.array([12, 13, 14],
                                              dtype='float64'),
                                'Z': pd.Categorical(
                                    ["fish", "chicken", "chicken"])})

        subdict = {'sub1': {"table": df_sub1,
                            "group_key": 'X',
                            "link_key": 'A'},
                   'sub2': {"table": df_sub2,
                            "group_key": 'X',
                            "link_key": 'E'}}

        self.df = df_main
        self.df_sub1 = df_sub1
        self.subdict = subdict

    def test_fit_transform(self):
        ac1 = AutoConverter(target='Target')
        X1, y1 = ac1.fit_transform(self.df, self.subdict)
        self.assertEqual(X1.shape[1],
                         len(ac1.get_feature_names()),
                         "# of features == # of feature names")
        self.assertIsInstance(X1, np.ndarray, 'wrong class returned')
        self.assertIsInstance(y1, np.ndarray, 'wrong class returned')

        ac1_f = AutoConverter(feature_only=True)
        ac1_l = AutoConverter(target='Target', label_only=True)

        X1a = ac1_f.fit_transform(self.df.drop("Target", axis=1), self.subdict)
        y1a = ac1_l.fit_transform(self.df, self.subdict)
        self.assertEqual(X1.shape, X1a.shape)
        self.assertEqual((y1 == y1a).sum(), len(y1))

        with self.assertRaises(ValueError):
            AutoConverter(feature_only=True, label_only=True)
        with self.assertRaises(ValueError):
            AutoConverter(label_only=True)  # target=None

        # Make sure "E" is considered categorical column
        ac1b = AutoConverter(target="Target",
                             coltype_dict={"E": "categorical"})
        X1b, y1b = ac1b.fit_transform(self.df)
        self.assertEqual(X1b.shape[1],
                         len(ac1b.get_feature_names()),
                         "# of features == # of feature names")

        self.assertEqual(X1.shape, (4, 17), "incorrect dataframe shape")
        temp_df = self.df_sub1[self.df_sub1['X'] == 2]

        # testing NumericalAggregator
        mean = temp_df['Y'].mean()
        std = temp_df['Y'].std()
        sum_ = temp_df['Y'].sum()
        count_ = temp_df['Y'].count()

        # this strange trick is used to 'align' dictionary orders in different
        # python versions
        fundict = {'sum': sum_, 'mean': mean, 'std': std, 'count': count_}
        for i, key in enumerate(sorted(fundict.keys())):
            self.assertEqual(X1[1, 9 + i], fundict[key],
                             "incorrect grouping on aggr function")

        # testing CategoryAggregator is impossible right here, since we need
        # 200+ items in test data to get 2 categories

        # testing correct work of AC with 'empty' tables:
        empty_df = pd.read_csv(
            'data/kaggle-kkbox-churn-prediction-challenge-1k/train.csv')
        # this should raise an exception
        ac2 = AutoConverter(target='is_churn')
        # A: 'empty' + nothing
        self.assertRaises(ValueError, ac2.fit_transform, empty_df)
        # B: 'empty' + something that does not join correctly
        # B1: wrong keys
        self.assertRaises(KeyError, ac2.fit_transform, empty_df, self.subdict)
        # B2: data mismatch (keys are okay, but data-wise it appears empty)
        # B2 does not produce any exceptions, just the zero matrix
        # C: 'empty' + 'empty' --> zero matrix and no exceptions

        # Test cases for manual configuration of column types
        titanic_df = pd.read_csv("data/train.csv")
        ac3 = AutoConverter(target="Survived")
        X3, y3 = ac3.fit_transform(titanic_df)
        self.assertEqual(ac3.colname_type_dict["Name"],
                         "text")
        self.assertIsInstance(X3, np.ndarray, 'wrong class returned')
        self.assertIsInstance(y3, np.ndarray, 'wrong class returned')

        with self.assertRaises(ValueError):
            AutoConverter(target="Survived",
                          coltype_dict={"Name": "something"})

        ac3a = AutoConverter(target="Survived",
                             coltype_dict={"Name": "categorical"})
        X3a, y3a = ac3a.fit_transform(titanic_df)
        self.assertEqual(ac3a.colname_type_dict["Name"],
                         "categorical")

        ac4 = AutoConverter(target="Target",
                            coltype_dict={"sub1": {"X": "categorical"}})
        # TODO(Yoshi):
        # Use the line below for ML-243
        #                                 "sub2": {"Y": "ignore"}})

        X4, y4 = ac4.fit_transform(self.df,
                                   self.subdict)
        self.assertEqual(ac4.colname_type_dict["sub1"]["X"],
                         "categorical")

        ac5 = AutoConverter(target="Survived", task_type='regression')
        X5, y5 = ac5.fit_transform(titanic_df)
        self.assertIsInstance(X5, np.ndarray, 'wrong class returned')
        self.assertIsInstance(y5, np.ndarray, 'wrong class returned')

        # TODO(Yoshi):
        # Same above. Use the line below for ML-243
        # self.assertEqual(ac4.colname_type_dict["sub1"]["Y"],
        #                 "ignore")

    def test_transform(self):
        # Test transform(df, prediction=True)
        ac = AutoConverter(target='Target')
        X, y = ac.fit_transform(self.df, self.subdict)
        # ac assumes "subtables" as input so it should raise AssertionError
        with self.assertRaises(AssertionError):
            ac.transform(self.df,
                         prediction=True)

        filtered_df = self.df.dropna(subset=["Target"])

        X_test = ac.transform(filtered_df,
                              subtables=self.subdict,
                              prediction=True)

        self.assertTrue(np.array_equal(X, X_test))

    def test_get_feature_names(self):
        ac = AutoConverter(target='Target')
        X, y = ac.fit_transform(self.df, self.subdict)

        feature_names = ac.get_feature_names()
        self.assertEqual(len(feature_names), X.shape[1],
                         "feature_names shape mismatch")
        self.assertEqual(feature_names[1], "main..B.date__Year")
        self.assertEqual(feature_names[9], "sub1..Y.numerical__count")

        self.assertEqual(len(ac.get_feature_names("A")),
                         1, "# of main..A features == 1")
        self.assertEqual(len(ac.get_feature_names(["sub1", "Y"])),
                         4, "# of sub1..Y features == 4")

    def test_save_load(self):
        """Call save() and load() functions."""
        ac = AutoConverter(target='Target')
        X, y = ac.fit_transform(self.df, self.subdict)
        tempdir = "__tmp__test__"
        filename = "ac_test.pickle"
        filepath = os.path.join(tempdir, filename)
        if os.path.exists(tempdir):
            shutil.rmtree(tempdir)
        os.makedirs(tempdir)
        self.assertTrue(ac.save(filepath))
        self.assertTrue(os.path.exists(filepath))
        self.assertFalse(ac.save(filepath,
                                 overwrite=False))
        loaded_ac = AutoConverter.load(filepath)
        self.assertEqual(ac.feature_names,
                         loaded_ac.feature_names)
        shutil.rmtree(tempdir)

    def test_task_type(self):
        """Tests correct work with classification/regression types."""

        self.df['Target'][4] = 42
        ac = AutoConverter(target='Target', task_type='classification')
        X, y = ac.fit_transform(self.df, self.subdict)
        self.assertEqual(y[4], 2)

        ac = AutoConverter(target='Target', task_type='regression')
        X, y = ac.fit_transform(self.df, self.subdict)
        print(y[4])
        self.assertEqual(y[4], 42)


class TextualAggregatorTestCase(unittest.TestCase):
    def setUp(self):
        self.subdf = pd.DataFrame([["A", "I went for a walk.",
                                    "私は散歩した"],
                                   ["A", "Stayed at home.", "家にいた"],
                                   ["B", "A", "私は猫である"],
                                   ["B", "Great", "すごい"],
                                   ["A", "Slept over.", "寝過ごした"]],
                                  columns=["uid", "text", "text_ja"])

    def test_fit_transform(self):
        # TODO(Yoshi): We might want to make group_key, colname interchangeable
        # regarding the order of arguments.
        self.assertRaises(TypeError,
                          TextualAggregator(group_key="uid", colname="text"))

        textual_agg0 = TextualAggregator(colname="text", group_key="uid")
        textual_ja_agg0 = TextualAggregator(colname="text_ja",
                                            group_key="uid")
        X = textual_agg0.fit_transform(self.subdf)
        X_ja = textual_ja_agg0.fit_transform(self.subdf)
        self.assertEqual(X.shape, (2, 9))
        self.assertEqual(len(textual_agg0.get_feature_names()), 9)
        self.assertEqual(X_ja.shape, (2, 5))
        self.assertEqual(len(textual_ja_agg0.get_feature_names()), 5)

        v1 = TfidfVectorizer(max_features=5)
        v2 = CountVectorizer(max_features=5)

        agg1 = TextualAggregator(colname="text", group_key="uid",
                                 vectorizer=v1)
        agg2 = TextualAggregator(colname="text", group_key="uid",
                                 vectorizer=v2)
        ja_vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 3))
        agg1_ja = TextualAggregator(colname="text_ja", group_key="uid",
                                    vectorizer=ja_vec)

        X1 = agg1.fit_transform(self.subdf)
        X2 = agg2.fit_transform(self.subdf)
        X1_ja = agg1_ja.fit_transform(self.subdf)
        self.assertEqual(X1.shape, X2.shape)
        self.assertEqual(agg1.get_feature_names(),
                         agg2.get_feature_names())
        self.assertEqual(X1_ja.shape, (2, 46))
        self.assertEqual(len(agg1_ja.get_feature_names()), 46)

        # Make sure there is no feature where all rows take 0
        self.assertEqual((X1.sum(axis=0) > 0).sum(),
                         len(agg1.get_feature_names()))
        self.assertEqual((X2.sum(axis=0) > 0).sum(),
                         len(agg2.get_feature_names()))

        # Output result consistency
        df2 = pd.DataFrame([[1, 1, 0, 1, 1], [0, 0, 1, 0, 0]])
        df2.index = ['A', 'B']
        df2.index.name = "uid"
        self.assertTrue(df2.equals(X2))

        v3 = CountVectorizer(max_features=None,
                             token_pattern=r"(?u)\b\w+\b")
        v4 = CountVectorizer(max_features=1,
                             token_pattern=r"(?u)\b\w+\b")
        agg3 = TextualAggregator(colname="text", group_key="uid",
                                 vectorizer=v3)
        agg4 = TextualAggregator(colname="text", group_key="uid",
                                 vectorizer=v4)
        X3 = agg3.fit_transform(self.subdf)
        X4 = agg4.fit_transform(self.subdf)
        self.assertEqual(X3.shape, (2, 11))
        self.assertEqual(X4.shape, (2, 1))

        # Make sure there is no feature where all rows take 0
        self.assertEqual((X3.sum(axis=0) > 0).sum(),
                         len(agg3.get_feature_names()))
        self.assertEqual((X4.sum(axis=0) > 0).sum(),
                         len(agg4.get_feature_names()))


class CategoryAggregatorTestCase(unittest.TestCase):
    """Due to the nature of parent code it's easier to test it separately."""
    def setUp(self):
        self.df = pd.DataFrame({'N': np.array([1, 2, 3, 17, 22, 1, 2, 3, 3]),
                                'Text': pd.Categorical(
                                    ["test", "train", "test", "train", "x",
                                     "y", "test", "train", "x"])})

    def test_fit(self):
        # testing that it returns category_aggr
        ca = CategoryAggregator(colname='Text', group_key='N')
        ca.fit(self.df)
        self.assertEqual(len(ca.get_feature_names()), 4,
                         'wrong dictionary fit')

        self.assertIsInstance(ca, CategoryAggregator, 'wrong class instance')

        ca1 = CategoryAggregator(colname='N', group_key='Text')
        ca1.fit(self.df)
        self.assertEqual(len(ca1.get_feature_names()), 5,
                         'wrong dictionary fit')

    def test_transform(self):
        ca = CategoryAggregator(colname='Text', group_key='N')
        ca.fit(self.df)
        X = ca.transform(self.df)
        self.assertIsInstance(X, pd.DataFrame, 'wrong return type')
        self.assertEqual(X.shape, (5, 4))
        self.assertEqual(X.iat[0, 1], 0)
        self.assertEqual(X.iat[3, 1], 1)


class TypeColumnTestCase(unittest.TestCase):
    def setUp(self):
        # check if test patterns in the directory
        self.dirpath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'testdata')
        self.assertTrue(os.path.exists(self.dirpath))

    def test_type_column(self):
        for filename in os.listdir(self.dirpath):
            filepath = os.path.join(self.dirpath,
                                    filename)
            info_list = filename.split("__")
            if len(info_list) != 2:
                # Invalid filename. Skip
                continue
            true_label = info_list[0]
            s = pd.read_csv(filepath,
                            squeeze=True, header=None)

            pred_label = type_column(s)
            if pred_label == true_label:
                flag = "OK"
            else:
                flag = "NG"
            print("[{}] Filename: {} True={} Pred={}".format(
                flag, filename, true_label, pred_label))
            self.assertEqual(true_label, pred_label)


class Python2EncodingTestCase(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([[1, u"カープ"],
                                [2, u"カープ"],
                                [3, u"レッドソックス"]])
        self.df.columns = [u"ターゲット", u"カテゴリ"]

    def test_autoconverter(self):
        ac = AutoConverter(target=u"ターゲット",
                           column_converters={
                               u"カテゴリ": [(TfidfVectorizer, {})]})
        X, _ = ac.fit_transform(self.df)
        self.assertEqual(X.shape, (3, 2))
