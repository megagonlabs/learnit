import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

from learnit.autoconverter.autoconverter import AutoConverter
from learnit.autolearn.autolearn import AutoLearn

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    ac = AutoConverter(target='Survived')
    X, y = ac.fit_transform(df)
    # ClassifierCatalog.level1, level2, level3, level4
    al = AutoLearn(level=2,
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
