import pandas as pd
from pandas import DataFrame
from plotnine import *
from preprocess_fllght_data import preprocess_flight_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV
import numpy as np
def linear(X_train, y_train, X_test, y_test):
    model_name = 'simple Linear Regression'
    reg = LinearRegression().fit(X_train, y_train)
    score_train = reg.score(X_train, y_train)
    score_test = reg.score(X_test, y_test)
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    EVS = explained_variance_score(y_test, y_pred)
    return model_name, y_pred, r2, MSE, score_train, score_test, EVS


# def Multiple_linear(X_train,y_train,X_test,y_test):
#     model_name = 'Multiple Linear Regression'
#     reg = LinearRegression()
#     slr_model=Pipeline(steps=[('preprocessorAll', X_train), ('regressor', reg)])
#     return model_name, y_pred, r2, MSE


def lasso_regression(X_train, y_train, X_test, y_test, lam):
    model_name = 'Lasso regression'
    lasso = Lasso(alpha=lam, normalize=True).fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    score_train = lasso.score(X_train, y_train)
    score_test = lasso.score(X_test, y_test)
    r2 = r2_score(y_test, y_pred)
    EVS = explained_variance_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)

    right_pred = np.where(np.sign(y_test) == np.sign(y_pred), 1, 0).sum() / len(y_test)
    print('right_pred', right_pred)

    print("save lasso results to file")
    pd.DataFrame(y_pred).to_csv("../pickle/lasso_pred.csv")

    return model_name, y_pred, r2, MSE, score_train, score_test, EVS


def Polynomial_linear(X_train, y_train, X_test, y_test, degree):
    model_name = 'Polynomial Regression'
    poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())  # k can be change
    poly.fit(X_train, y_train)
    score_train = poly.score(X_train, y_train)
    score_test = poly.score(X_test, y_test)
    y_pred = poly.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    EVS = explained_variance_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    return model_name, y_pred, r2, MSE, score_train, score_test, EVS


def Decision_trees(X_train, y_train, X_test, y_test):
    model_name = 'Decision trees'
    regressor = DecisionTreeRegressor(max_features=0.95, min_samples_split=0.2, min_samples_leaf=0.05)
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


def Elastic_cv(X_train, y_train, X_test, y_test):
    model_name = 'elastic trees'
    # regressor = DecisionTreeRegressor(max_features=0.95, min_samples_split=0.2, min_samples_leaf=0.05)
    # elastic = ElasticNetCV(normalize=True, cv=5)
    elastic = ElasticNetCV(l1_ratio=0.2, cv=10,max_iter=10000,random_state=1000)
    # search=GridSearchCV(estimator=elastic,param_grid={'alpha':np.logspace(-5,2,8),'l1_ratio':[.2,.4,.6,.8]},
    #                     scoring='neg_mean_squared_error',n_jobs=1,refit=True,cv=10)
    elastic.fit(X_train, y_train)
    score_train = elastic.score(X_train, y_train)
    score_test = elastic.score(X_test, y_test)
    y_pred = elastic.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    EVS = explained_variance_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)

    right_pred = np.where(np.sign(y_test) == np.sign(y_pred), 1, 0).sum() / len(y_test)
    print('right_pred', right_pred)

    print("save elastic trees to file")
    pd.DataFrame(y_pred).to_csv("../pickle/elastic_tree_pred.csv")

    return model_name, y_pred, r2, MSE, score_train, score_test, EVS


def Random_forest_trees(X_train, y_train, X_test, y_test):
    model_name = 'Random forest'
    forest = RandomForestRegressor(n_estimators=100, max_depth=75, min_samples_leaf=0.01, random_state=0)
    forest.fit(X_train, y_train)
    score_train = forest.score(X_train, y_train)
    score_test = forest.score(X_test, y_test)
    y_pred = forest.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    EVS = explained_variance_score(y_test, y_pred)
    return model_name, y_pred, r2, MSE, score_train, score_test, EVS


def get_best_reg_model(X_train, y_train, X_test, y_test):
    dfs = []

    print("Decision trees")
    model_name, y_pred, r2, MSE, score_train, score_test, EVS = Decision_trees(X_train, y_train,
                                                                               X_test, y_test)
    dfs.append(DataFrame(
        {'model_name': model_name, 'r2': r2, 'MSE': MSE, 'score_train': score_train,
         'score_test': score_test, 'EVS': EVS}, index=[0]))

    # print("Linear")
    # model_name, y_pred, r2, MSE, score_train, score_test, EVS = linear(X_train, y_train, X_test,
    #                                                                    y_test)
    # dfs.append(DataFrame(
    #     {'model_name': model_name, 'r2': r2, 'MSE': MSE, 'score_train': score_train,
    #      'score_test': score_test, 'EVS': EVS}, index=[0]))

    # print("Polynomial linear")
    # model_name, y_pred, r2, MSE, score_train, score_test, EVS = Polynomial_linear(X_train, y_train,
    #                                                                               X_test, y_test,
    #                                                                               degree=3)
    # dfs.append(DataFrame(
    #     {'model_name': model_name, 'r2': r2, 'MSE': MSE, 'score_train': score_train,
    #      'score_test': score_test, 'EVS': EVS}))

    # print("Elastic_cv")
    # model_name, y_pred, r2, MSE, score_train, score_test, EVS = Elastic_cv(X_train, y_train,
    #                                                                            X_test, y_test)
    # dfs.append(DataFrame(
    #     {'model_name': model_name, 'r2': r2, 'MSE': MSE, 'score_train': score_train,
    #      'score_test': score_test, 'EVS': EVS}, index=[0]))

    # print("Random forest trees")
    # model_name, y_pred, r2, MSE, score_train, score_test, EVS = Random_forest_trees(X_train,
    #                                                                                 y_train, X_test,
    #                                                                                 y_test)
    # dfs.append(DataFrame(
    #     {'model_name': model_name, 'r2': r2, 'MSE': MSE, 'score_train': score_train,
    #      'score_test': score_test, 'EVS': EVS}, index=[0]))

    # print("Lasso regression")
    # model_name, y_pred, r2, MSE, score_train, score_test, EVS = lasso_regression(X_train, y_train,
    #                                                                              X_test, y_test,
    #                                                                              lam=.0005)
    # dfs.append(DataFrame(
    #     {'model_name': model_name, 'r2': r2, 'MSE': MSE, 'score_train': score_train,
    #      'score_test': score_test, 'EVS': EVS}, index=[0]))

    dfs = pd.concat(dfs)
    return dfs


