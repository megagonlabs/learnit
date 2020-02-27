import pandas as pd
from tabulate import tabulate

from learnit.autoconverter.autoconverter import AutoConverter
from learnit.autolearn.autolearn import AutoLearn
# from learnit.autolearn.evaluate import Evaluate

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    # Read the dataset to Pandas DataFrame
    # pd.read_csv('path to your dataset')
    #     parse_parameter is vital for correct reading of DateTime columns
    #     of the dataset: list them as shown
    data_path = 'data/kaggle-kkbox-churn-prediction-challenge-1k/'
    df = pd.read_csv(data_path + 'members_train.csv',
                     parse_dates=['registration_init_time', 'expiration_date'])

    # Create AutoConverter object used for conversion of data table to feature
    # matrix.
    #     target parameter denotes the target column name in the dataset.
    ac = AutoConverter(target='is_churn')

    # Fit the transformer and convert(transform) the table to the feature
    # matrix X and target vector y
    X, y = ac.fit_transform(df)

    # Create the basic AutoLearn Object
    al = AutoLearn(level=1)

    # train the classifier
    results = al.learn(X, y)

    al.display()

    # Prediction phase---------------------------------------------------------
    # The results of calculating the error on train-test splits are indicated
    # above. The following procedure is for predicting on unseen dataset:
    # open the new data for which you need a prediction to be made:
    df_unseen = pd.read_csv(data_path + 'members_test.csv',
                     parse_dates=['registration_init_time', 'expiration_date'])

    # Transform this new dataset to the feature matrix. Note that you do
    # not fit, only transform this time, since transformer is already fitted
    X_predict = ac.transform(df_unseen, prediction=True)

    # Predict and decode the labels
    prediction = ac.index2label(al.predict(X_predict))
    df_unseen['prediction'] = prediction
    print(tabulate(df_unseen[['prediction', 'is_churn']],
        headers='keys', tablefmt='psql'))

    # Todo(Kate):
    # add tutorials for results evaluation here'''

