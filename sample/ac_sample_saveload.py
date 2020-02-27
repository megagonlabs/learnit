import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from learnit.autoconverter.autoconverter import AutoConverter
from learnit.autolearn.autolearn import AutoLearn
from learnit.autolearn.evaluate import Evaluate
from learnit.autolearn.blueprints import StackedXGBoost, AverageBlender

if __name__ == '__main__':

    ac_filepath = "tmp/sample_ac.pickle"
    al_filepath = "tmp/sample_al.pickle"
    df = pd.read_csv('data/train.csv')

    if not (os.path.exists(ac_filepath) and os.path.exists(al_filepath)):
        ac = AutoConverter(target='Survived')
        X, y = ac.fit_transform(df)
        al = AutoLearn(customized_clf_list=[('LogisticRegression',
                                             LogisticRegression())],
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

        ac.save(ac_filepath)
        al.save(al_filepath)
    
    ac = AutoConverter.load(ac_filepath)
    al = AutoLearn.load(al_filepath)

    e = Evaluate(ac, al)
    orig_eval_s = e.evaluate_performance(df)
    col_imp_df = e.calculate_column_importance(df)
