
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from preprocess_fllght_data import preprocess_flight_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def score(model,X_test ,y_test,model_name):
    y_pred = model.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    pre=precision_score(y_test, y_pred, average='weighted')
    rec=recall_score(y_test, y_pred, average='weighted')
    f1=f1_score(y_test, y_pred, average='weighted')
    return pd.DataFrame({'model_name':model_name,'accuracy':acc,'precision':pre,'recall':rec,'f1':f1}, index=[0])

def SVM(X_train,y_train,X_test,y_test):
    svm = SVC(C=1e6, kernel='linear',decision_function_shape='ovo')
    svm.fit(X_train,y_train)
    s=score(svm,X_test, y_test)
    return s

def Logistic(X_train,y_train,X_test,y_test):
    model_name='Logistic'
    logistic = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter = 80)
    logistic.fit(X_train,y_train)
    s=score(logistic,X_test, y_test,model_name)
    return s

# 7. e.
def DecisionTree(X_train,y_train,X_test,y_test):
    model_name='DecisionTree'
    tree = DecisionTreeClassifier(min_samples_split=0.1)
    tree.fit(X_train,y_train)
    s=score(tree,X_test, y_test,model_name)
    return s

def RandomForest(X_train, y_train, X_test, y_test):
    model_name='RandomForest'
    randtree = RandomForestClassifier(n_estimators=100,min_samples_split=0.1, random_state=0)
    randtree.fit(X_train,y_train)
    s=score(randtree,X_test, y_test,model_name)
    return s

def  Soft_SVM(X_train,y_train,X_test,y_test):
    model_name='Soft_SVM'
    s_svm = SVC(C=1e-2, kernel='poly',decision_function_shape='ovr')
    s_svm.fit(X_train,y_train)
    s=score(s_svm,X_test, y_test,model_name)
    return s


def k_nearest_neighbors(X_train,y_train,X_test,y_test):
    model_name='k_nearest_neighbors'
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train,y_train)
    s=score(knn,X_test, y_test,model_name)
    return s

def nn(X_train,y_train,X_test,y_test):
    model_name='neural_network'
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state = 1)
    clf.fit(X_train,y_train)
    s=score(clf,X_test, y_test,model_name)
    return s


if __name__ == "__main__":
    df = pd.read_csv("../data/data1k.csv", dtype={'FlightDate': str, 'CRSDepTime': str,
                                                  'CRSArrTime': str})
    X, y_delay, y_type = preprocess_flight_data(df)
    listNAN=list(y_type[y_type.isna()].index)
    X=X.drop(listNAN)
    y_type=y_type.drop(listNAN)
    X_train, X_test, y_train, y_test = train_test_split(X, y_type, test_size=0.2)
    #dfs=[]
    #dfs=SVM(X_train,y_train,X_test,y_test)
    dfs =Logistic(X_train, y_train, X_test, y_test)
    dfs=dfs.append(DecisionTree(X_train, y_train, X_test, y_test),ignore_index=True)
    dfs=dfs.append(RandomForest(X_train, y_train, X_test, y_test),ignore_index=True)
    dfs=dfs.append(Soft_SVM(X_train, y_train, X_test, y_test))
    dfs=dfs.append(k_nearest_neighbors(X_train, y_train, X_test, y_test),ignore_index=True)
    dfs=dfs.append(nn(X_train, y_train, X_test, y_test),ignore_index=True)
    o=0