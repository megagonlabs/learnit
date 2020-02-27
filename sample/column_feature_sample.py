import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from learnit.autoconverter.autoconverter import AutoConverter
from learnit.autolearn.autolearn import AutoLearn
from learnit.autolearn.evaluate import Evaluate
from learnit.autolearn.blueprints import StackedXGBoost, AverageBlender

from sklearn.base import BaseEstimator, TransformerMixin


class TitleEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 title_list=None):
        if title_list is None:
            title_list = ["Mr.",
                          "Miss.",
                          "Mrs.",
                          "Master."]
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


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    ac = AutoConverter(target='Survived',
                       column_converters={"Name": [(TitleEncoder, {})],
                                          "Pclass": [],
                                          "Sex": [],
                                          "Age": [],
                                          "SibSp": [],
                                          "Parch": [],
                                          "Ticket": [],
                                          "Fare": [],
                                          "Cabin": [],
                                          "Embarked": []})
    X, y = ac.fit_transform(df)
    al = AutoLearn(customized_clf_list=[('LogisticRegression', LogisticRegression())],
                   metric='roc_auc',
                   cv_num=5,
                   pos_label=1,
                   n_jobs=1,
                   verbose=0)
    results = al.learn(X, y)
    print(results['name'])
    print(results['eval_df'])

    pred = al.predict(X)
    print(pred)

    # name, clf, cv_result = autolearn(X, y, verbose=3, clf_list=[('AverageBlender', AverageBlender(scoring='roc_auc', random_state=1, verbose=3))])
    #name, clf, cv_result = autolearn(X, y, verbose=3, clf_list=[('LogisticRegression', LogisticRegression())])

    e = Evaluate(ac, al)
    orig_eval_s = e.evaluate_performance()
    col_imp_df = e.calculate_column_importance()
    explain_df = e.get_explanations(df)
    
    X_test = ac.transform(df, prediction=True)
    