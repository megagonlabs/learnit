# Feature Engineering with Learn-It


## User-defined Transformer

A transformer should inherit `BaseEstimator` and `TransformerMixin`.

You can implement a new feature in the form of Python class. The class has the following
requirements.

- It must implement `transform()`
- (If necessary)
    - (If it needs to fit to training data) it should implement `fit()`
    - (If it needs to do something in the initialization phase,) it should implement `__init__()`
- (Optional) `get_feature_names()`


Here is an example of a transformer that extracts a 0/1-encoded vector based on
 the title (e.g., Mr., Ms. etc.) information from a textual column that contains names.

```
from sklearn.base import BaseEstimator, TransformerMixin

class TitleEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 title_list=None):
        if title_list is None:
            title_list = ["Mr.",
                          "Ms."
                          "Mrs.",
                          "Miss."]
        self.title_list = title_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is np.array
        s = pd.Series(X[:, 0])
        s_list = []
        for title in self.title_list:
            s_list.append(s.apply(lambda x: title in x).astype("int"))
        df = pd.concat(s_list, axis=1)
        df.loc[:, "None"] = (df.sum(axis=1) == 0).astype("int")
        return df.as_matrix()

    def get_feature_names(self):
        return self.title_list + ["Others"]
```

### `__init__()`

This `__init__` function receives `title_list` so that the user can customize
the title list. For instance, someone might want to incorporate "Prof." and
"Dr." in addition to the common titles. In this case, the four titles will be
used by default.


### `fit()`

This example does nothing in the training phase. The function must return
`self` by definition.


### `transform()`

This is the main function of the feature transformer class. This example is
assigning

As you see, the variable `X` is `np.ndarray` of the shape `(N, 1)` where `N`
is the total number of rows in the input data.


In the following example, `AutoConverter` instance will use `TitleEncoder` transformer
 to the "Name" column in the data for feature extraction while automatically
 selected transformers will be applied to the other columns.

```
>>> df = pd.read_csv('data/train.csv')
>>> ac = AutoConverter(target='Survived',
                       column_converters={"Name": [(TitleEncoder, {})]})
```

Note that the automatically assigned transformers will not be applied to
  a column if the user specifies transformer(s) to the column. If you want to
  apply default transformers to the column type in addition to customized transformer(s),
  use `use_column_converter_only=False` option (it is `True` by default.)

```
>>> df = pd.read_csv('data/train.csv')
>>> ac = AutoConverter(target='Survived',
                       column_converters={"Name": [(TitleEncoder, {})]},
                       use_column_converter_only=False)
```
