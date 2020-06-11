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
    regressor = DecisionTreeRegressor(random_state=0, max_features=0.5)
    regressor.fit(X_train, y_train)
    score_train = regressor.score(X_train, y_train)
    score_test = regressor.score(X_test, y_test)
    y_pred = regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    EVS = explained_variance_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    return model_name, y_pred, r2, MSE, score_train, score_test, EVS


def Random_forest_trees(X_train, y_train, X_test, y_test):
    model_name = 'Random forest'
    forest = RandomForestRegressor(n_estimators=100, max_depth=80, min_samples_leaf=0.005, random_state=0)
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

    # print("Linear")
    # model_name, y_pred, r2, MSE, score_train, score_test, EVS = linear(X_train, y_train, X_test,
    #                                                                    y_test)
    # dfs.append(DataFrame(
    #     {'model_name': model_name, 'r2': r2, 'MSE': MSE, 'score_train': score_train,
    #      'score_test': score_test, 'EVS': EVS}, index=[0]))

    print("Polynomial linear")
    # model_name, y_pred, r2, MSE, score_train, score_test, EVS = Polynomial_linear(X_train, y_train,
    #                                                                               X_test, y_test,
    #                                                                               degree=3)
    # dfs.append(DataFrame(
    #     {'model_name': model_name, 'r2': r2, 'MSE': MSE, 'score_train': score_train,
    #      'score_test': score_test, 'EVS': EVS}))

    print("Decision trees")
    model_name, y_pred, r2, MSE, score_train, score_test, EVS = Decision_trees(X_train, y_train,
                                                                               X_test, y_test)
    dfs.append(DataFrame(
        {'model_name': model_name, 'r2': r2, 'MSE': MSE, 'score_train': score_train,
         'score_test': score_test, 'EVS': EVS}, index=[0]))

    print("Random forest trees")
    model_name, y_pred, r2, MSE, score_train, score_test, EVS = Random_forest_trees(X_train,
                                                                                    y_train, X_test,
                                                                                    y_test)
    dfs.append(DataFrame(
        {'model_name': model_name, 'r2': r2, 'MSE': MSE, 'score_train': score_train,
         'score_test': score_test, 'EVS': EVS}, index=[0]))

    print("Lasso regression")
    model_name, y_pred, r2, MSE, score_train, score_test, EVS = lasso_regression(X_train, y_train,
                                                                                 X_test, y_test,
                                                                                 lam=.5)
    dfs.append(DataFrame(
        {'model_name': model_name, 'r2': r2, 'MSE': MSE, 'score_train': score_train,
         'score_test': score_test, 'EVS': EVS}, index=[0]))

    dfs = pd.concat(dfs)
    return dfs

# explained_variance_score

