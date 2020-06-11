import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.tree import DecisionTreeRegressor


def decision_tree(X_train, y_train, X_test, y_test):
    model_name = 'Decision trees'
    decision_tree_model = DecisionTreeRegressor(max_features=0.8, min_samples_split=0.2, min_samples_leaf=0.05)
    decision_tree_model.fit(X_train, y_train)
    score_train = decision_tree_model.score(X_train, y_train)
    score_test = decision_tree_model.score(X_test, y_test)
    y_pred = decision_tree_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    right_pred = np.where(np.sign(y_test) == np.sign(y_pred), 1, 0).sum() / len(y_test)
    print('right_pred', right_pred)

    print("save decision trees results to file")
    score = DataFrame({'model_name': model_name, 'r2': r2, 'MSE': mse, 'score_train': score_train, 'score_test':
        score_test, 'EVS': evs}, index=[0])
    return decision_tree_model, score


def get_reg_model(X_train, y_train, X_test, y_test):
    print("Decision trees")
    return decision_tree(X_train, y_train, X_test, y_test)
