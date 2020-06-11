import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def score(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return pd.DataFrame(
        {'model_name': model_name, 'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1},
        index=[0])


def RandomForest(X_train, y_train, X_test, y_test):
    model_name = 'RandomForest'
    randtree = RandomForestClassifier(n_estimators=100, min_samples_split=0.01, random_state=0)
    randtree.fit(X_train, y_train)
    s = score(randtree, X_test, y_test, model_name)
    return s


def get_classification_model(X_train, y_train, X_test, y_test):
    return RandomForest(X_train, y_train, X_test, y_test)
