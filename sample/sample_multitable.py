"""
AutoConverter with multiple tables example.

In many situations you may need to use multiple data tables. The following
method takes care of it. After you perform conversion you can proceed with
AutoLearn as usual.
"""

import pandas as pd
from learnit.autoconverter.autoconverter import AutoConverter
from learnit.autolearn.autolearn import AutoLearn


# Read the dataset to Pandas DataFrame
# pd.read_csv('path to your dataset')
#     parse_parameter is vital for correct reading of DateTime columns
#     of the dataset: list them as shown
# main dataset that contains target value
dataset_folder = 'data/kaggle-kkbox-churn-prediction-challenge-1k/'
df_main = pd.read_csv(dataset_folder + 'members_train.csv', parse_dates=[
    'registration_init_time', 'expiration_date'])
# secondary dataset that needs to be linked to the main one
df_sub = pd.read_csv(dataset_folder + 'transactions.csv', parse_dates=[
    'membership_expire_date', 'transaction_date'])

# Create AutoConverter object used for conversion of data table to feature
# matrix.
#     target parameter denotes the target column name in the dataset.
ac = AutoConverter(target='is_churn')

# the following dictionary is used to tell AutoConverter object how to organize
# and link the secondary tables. It can contain more than one table
#       'table' - name of the DataFrame that we just loaded
#       'link_key' - name of the column in the main table that is used to link
#   to this particular table
#       'group_key' - name of the column in the secondary table that is used
#   for linking and for grouping, since secondary table might contain more rows
#   then there are in the main one. Merging tables would be made using
#   main[link_key] = secondary[group_key]

subtables = {'second_table': {'table': df_sub,
                           'link_key': 'msno',
                           'group_key': 'msno'}}

# Fit the transformer and convert(transform) the table to the feature
# matrix X and target vector y
# subtables are passed as a parameter during this procedure
X, y = ac.fit_transform(df_main, subtables)

print(X.shape)


# now you can use X and y as usual and proceed to the main tutorial sample.py








