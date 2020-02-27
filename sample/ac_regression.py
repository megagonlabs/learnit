import pandas as pd

from learnit.autoconverter.autoconverter import AutoConverter
from learnit.autolearn.autolearn import AutoLearn
from learnit.autolearn.evaluate import Evaluate
from sklearn import datasets

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # For this tutorial we would use 'boston' dataset. For normal workflow,
    # including AutoConverter, pleae do the following
    # df = pd.read_csv('path to file')
    # ac = AutoConverter(target='Age', task_type='regression')
    # X, y = ac.fit_transform(df)
    # the only difference for AutoConverter is making task_type='regression'

    boston_data = datasets.load_boston()
    X = boston_data.data
    y = boston_data.target

    # dividing data to training and test data (for demonstration purposes)
    # for proper workflow refer to sample.py
    X_test = X[0:10]
    X_train = X[10:]
    y_test = y[0:10]
    y_train = y[10:]


    # Create the basic AutoLearn Object
    al = AutoLearn(level=1, task='regression')

    # train the classifier
    al.fit(X_train, y_train)

    al.display()

    # generating the result table (for demonstration purposes)
    prediction = al.predict(X_test)
    df_show = pd.DataFrame()
    df_show['prediction'] = prediction
    df_show['real'] = y_test

    print(df_show)


