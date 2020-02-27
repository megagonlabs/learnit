# Step-by-step tutorial

You can use `sample/sample.py` file as your starting point. Let's revise it step by step:

1. Import the modules that you will use.

```
>>> import pandas as pd
>>> from learnit.autoconverter.autoconverter import AutoConverter
>>> from learnit.autolearn.autolearn import AutoLearn
```

2. Load data from the input CSV file. LearnIt uses `pandas.DataFrame` as a base object. For date columns, 
list the names of the columns in `parse_dates` parameter.
If you have multiple tables, please read [this tutorial](/docs/ac_multitable.md).

```
>>> df = pd.read_csv('path to your .csv file', parse_dates=['AppointmentDate', 'DateTime-column-2'])
```
| AppointmentDate | Neighbourhood | Gender | Age | No-show (target) |
|-----------------|---------------|--------|-----|---------|
| 15/09/2017| Conquista| M | 55| Yes|
| 13/01/2016| Republica| F | 22| No |
| 01/01/2017| Conquista| F | 2 | Yes|

3. `AutoConverter` object is used for transformation of data into the feature matrix.
This is an important step that converts your data into the format that is used by machine learning algorithms.
Target column is the column of your table that you would like to predict. Let's first create this object:

```
>>> ac = AutoConverter(target='Target-column-name')
```

4. Convert the dataset to the feature matrix and the target vector by using the AutoConverter.
After this operation `ac` AutoConverter object learns the conversion rules and can transform other (unseen) data of the 
same format in the same manner as it does with the original dataset. `X` is the feature matrix and `y` is the 'target' vector.

```
>>> X, y = ac.fit_transform(df)
```
This is how feature matrix might look like (column names are approximate and given for your better understanding about 
what happened to the dataset). Autoconverter took care of all the values and converted them to numerical.

| Day | Month | Year | Weekday | ... | Conquista | Republica | ... | M | F | Age|...|
|-----|-------|------|---------|-----|-----------|-----------|-----|--|--|---|---|
|15|9|2017|...|...|1|0|...|1|0|55 | ...|
|13|1|2016|...|...|0|1|...|0|1|22 | ...|
|1|1|2017|...|...|1|0|...|0|1|2 | ...|

The target vector will look like `[1, 0, 1, ...]`

5. Create `AutoLearn` object that will handle the whole classification task for you. First you need to create the object
 and then pass the data to it, so that it could train on this data.
```
>>> al = AutoLearn()
>>> al.learn(X, y)
```
`al` object now contains the best trained model and some very useful functions for it. Let's print out some information 
about it using display function:
```
>>> al.display()
+-----------------------+------------------+
| metric                | value            |
|-----------------------+------------------|
| Model is trained      | True             |
| Best classifier       | XGBClassifier    |
| Evaluation metric     | roc_auc          |
| Dataset size          | 898              |
| # of features         | 26               |
| Classifier set level  | 2                |
| Including classifiers | ['GBClassifier'] |
| Training time         | 5.15 sec.        |
+-----------------------+------------------+
+-----------+----------------+------------+
| metric    |   training set |   test set |
|-----------+----------------+------------|
| Accuracy  |       0.978008 |   0.966623 |
| Precision |       0.981875 |   0.919498 |
| Recall    |       0.810924 |   0.761922 |
| roc_auc   |       0.970514 |   0.880481 |
+-----------+----------------+------------+
```

From those values you can get the idea of how well the model predicts the target values.
 Test error would be a good estimate of performance on new, unseen data.
  If the model does not perform well, you might want to try different level of `Autolearn(level=n)` where n = 2, 3, 4, 5 object
   in your code. Please note that raising the level means that you spend more time on training.

6. You can save your AutoConverter and AutoLearn objects to reuse them later:
```
>>> ac.save(ac_filepath)
>>> al.save(al_filepath)
```

### Let's predict using the trained model!

As soon as you trained the model, you can use it for predicting.

1. First we need to load our saved objects for converting and predictiong:
```
>>> ac = AutoConverter.load(ac_filepath)
>>> al = AutoLearn.load(al_filepath)
```
2. Let's read the table with new data for which we want to get the predictions:

```
>>> df_unseen = pd.read_csv('path to new data', parse_dates=['...', '...'])
```

3. We need to use `ac` object to create the feature matrix. We only use `transform` function of it, since it has already been fitted (It already learnt how it needs to convert the data during the learning phase). We also indicate our `AutoConverter` instance that we are in prediction phase and there won't be any target vector as a result of this operation.
```
>>> X_predict = ac.transform(df_unseen, prediction=True)
```

4. Now we can use this new feature matrix for predictions!
`al.predict()` is predicting the target labels, but those labels are still
encoded by `AutoConverter` to integers. To decode the original values back we use `index2label` function.
```
>>> prediction = ac.index2label(al.predict(X_predict))
>>> print(prediction)
  ```

Please note that prediction quality might not be ideal.

### Training the model for regression

The process is very similar to classification. The only difference is that regression is used for predicting numerical values instead of classifying an object to a certain class.

`AutoConverter` works the same way as with classification, but you specify `task-type=regression`.
```
>>> ac = AutoConverter(target='Target-column-name', task_type='regression')
```
Transform the data with the converter:

```
>>> X, y = ac.fit_transform(df)
```

Then we can create `AutoLearn` object the same way we did for classification, but we need to specify the `task=regression`

>>> al = AutoLearn(task='regression')
>>> al.learn(X, y)

To look at what we get as `al` we can use display function:

```
>>> al.display()
```
which returns
```
+----------------------+-------------------------+
| metric               | value                   |
|----------------------+-------------------------|
| Model is trained     | True                    |
| Best estimator       | Ridge                   |
| Evaluation metric    | neg_mean_absolute_error |
| Dataset size         | 496                     |
| # of features        | 13                      |
| Estimator set level  | 1                       |
| Including estimators | ['Ridge']               |
| Training time        | 1.21 sec.               |
+----------------------+-------------------------+
+-------------------------+----------------+------------+
| metric                  |   training set |   test set |
|-------------------------+----------------+------------|
| Mean abs error          |      -3.16993  |  -4.17899  |
| Mean sq error           |     -20.764    | -37.3245   |
| r2                      |       0.750799 |   0.354923 |
| neg_mean_absolute_error |      -3.16993  |  -4.17899  |
+-------------------------+----------------+------------+
```

You can do all the things you can do in case of classification in case of regression: save, load.
Please refer to the manual above.
