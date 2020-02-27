import os

import pandas as pd

from learnit.autoconverter.autoconverter import AutoConverter
from learnit.autolearn.autolearn import AutoLearn
from learnit.autolearn.evaluate import Evaluate

"""
subtables =
{tabname(str): {"table": (pd.Dataframe),
                "link_key": (str) main table column name,
                "group_key": (str) this table column name}}
"""

if __name__ == '__main__':
    dirpath = "data/kaggle-kkbox-churn-prediction-challenge-1k"
    main_df = pd.read_csv(os.path.join(dirpath, "train.csv"))
    members_df = pd.read_csv(os.path.join(dirpath, "members_train.csv"))
    transactions_df = pd.read_csv(os.path.join(dirpath, "transactions.csv"))
    user_logs_df = pd.read_csv(os.path.join(dirpath, "user_logs.csv"))

    subtables = {"transactions": {"table": transactions_df,
                                  "link_key": "msno",
                                  "group_key": "msno"},
                 "user_logs": {"table": user_logs_df,
                               "link_key": "msno",
                               "group_key": "msno"}}

    ac = AutoConverter(target="is_churn")
    X, y = ac.fit_transform(df=members_df,
                            subtables=subtables)
    al = AutoLearn(level=2)
    al.learn(X, y)
    e = Evaluate(ac=ac, alearn=al)
    orig_eval_s = e.evaluate_performance()
    col_imp_df = e.calculate_column_importance()