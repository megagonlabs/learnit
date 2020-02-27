import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from learnit.autoconverter.autoconverter import AutoConverter
from learnit.autolearn.autolearn import AutoLearn
from learnit.autolearn.evaluate import Evaluate
from learnit.autolearn.blueprints import StackedXGBoost, AverageBlender

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    ac = AutoConverter(target='Survived')
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
    