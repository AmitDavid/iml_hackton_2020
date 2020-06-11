import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from plotnine import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from preprocess_fllght_data import preprocess_flight_data
from pandas import DataFrame



def linear(X_train,y_train,X_test,y_test):
    model_name='simple Linear Regression'
    reg = LinearRegression().fit(X_train,y_train)
    r2=reg.score(X_train,y_train)
    coef=reg.coef_
    y_pred=reg.predict(X_test)
    MSE=mean_squared_error(y_test, y_pred)
    return model_name,y_pred,r2,MSE

# def Multiple_linear(X_train,y_train,X_test,y_test):
#     model_name = 'Multiple Linear Regression'
#     reg = LinearRegression()
#     slr_model=Pipeline(steps=[('preprocessorAll', X_train), ('regressor', reg)])
#     return model_name, y_pred, r2, MSE

def Polynomial_linear(X_train,y_train,X_test,y_test):
    model_name='polynomial Regression'
    poly = make_pipeline(PolynomialFeatures(3), LinearRegression()) # k can be change
    poly.fit(X_train,y_train)
    y_pred = poly.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    return model_name, y_pred, r2, MSE

def Decision_trees(X_train,y_train,X_test,y_test):
    model_name = 'decision trees Regression'
    regressor = DecisionTreeRegressor(random_state=0,max_features=10)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    return model_name, y_pred, r2, MSE

def Random_forest_trees(X_train,y_train,X_test,y_test):
    model_name = 'Random forest trees Regression'
    forest = RandomForestRegressor(n_estimators=100,max_depth=10, random_state=0)
    forest.fit(X_train,y_train)
    y_pred = forest.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    return model_name, y_pred, r2, MSE

if __name__ == '__main__':
    df = pd.read_csv("../data/data1k.csv", dtype={'FlightDate': str, 'CRSDepTime': str,
                                                  'CRSArrTime': str})
    X, y_delay, y_type = preprocess_flight_data(df)
    # X.to_csv("results_X.csv")
    # y.to_csv("results_y.csv")
    # # print('done')
    # listNAN=list(y_delay[y_delay.isna()].index)
    # X=X.drop(listNAN)
    # y_delay=y_delay.drop(listNAN).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y_delay, test_size=0.2)  # , random_state=42)
    dfs=[]
    model_name, y_pred, r2, MSE=linear(X_train,y_train,X_test,y_test)
    dfs.append(DataFrame({'model_name': model_name, 'r2': r2, 'MSE': MSE}, index=[0]))
    # model_name, y_pred, r2, MSE=Polynomial_linear(X_train,y_train,X_test,y_test)
    # dfs.append(DataFrame({'model_name': r"${}$".format(model_name), 'r2': r2, 'MSE':MSE}))
    model_name, y_pred, r2, MSE=Decision_trees(X_train,y_train,X_test,y_test)
    dfs.append(DataFrame({'model_name': model_name, 'r2': r2, 'MSE': MSE}, index=[0]))
    model_name, y_pred, r2, MSE=Random_forest_trees(X_train,y_train,X_test,y_test)
    dfs.append(DataFrame({'model_name': model_name, 'r2': r2, 'MSE': MSE}, index=[0]))
    dfs = pd.concat(dfs)
    ggplot(dfs) + geom_bar(aes(x='MSE'))
    t=5
# explained_variance_score