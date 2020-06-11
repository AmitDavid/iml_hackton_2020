import pandas as pd
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np


def Decision_trees(X_train, y_train, X_test, y_test):
    model_name = 'Decision trees'
    regressor = DecisionTreeRegressor(max_features=0.8, min_samples_split=0.2, min_samples_leaf=0.05)
    regressor.fit(X_train, y_train)
    score_train = regressor.score(X_train, y_train)
    score_test = regressor.score(X_test, y_test)
    y_pred = regressor.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    EVS = explained_variance_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)

    right_pred = np.where(np.sign(y_test) == np.sign(y_pred), 1, 0).sum() / len(y_test)
    print('right_pred', right_pred)

    print("save decision trees results to file")
    pd.DataFrame(y_pred).to_csv("../pickle/dec_tree_pred.csv")

    return model_name, y_pred, r2, MSE, score_train, score_test, EVS


def get_reg_model(X_train, y_train, X_test, y_test):
    print("Decision trees")
    model_name, y_pred, r2, MSE, score_train, score_test, EVS = Decision_trees(X_train, y_train, X_test, y_test)
    return DataFrame({'model_name': model_name, 'r2': r2, 'MSE': MSE, 'score_train': score_train, 'score_test':
        score_test, 'EVS': EVS}, index=[0])
