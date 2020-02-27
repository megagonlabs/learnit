from learnit.autolearn.autolearn import AutoLearn
from sklearn import datasets

if __name__ == '__main__':
    data = datasets.load_iris()
    X = data.data
    y = data.target

    al = AutoLearn()
    result_info = al.learn(X, y)

