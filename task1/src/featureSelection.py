# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, MinMaxScaler

from feature_selector import FeatureSelector
from sklearn.feature_selection import SelectFdr, f_classif, mutual_info_classif, SelectKBest
from scipy import stats

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import seaborn as sns

#np.seterr(divide='ignore', invalid='ignore')

def get_X_y_data():
    data_expression = pd.read_csv("expression_BRCA_TCGA.csv", header=0, index_col=0)
    data_expression = data_expression.T
    X = data_expression
    labels = pd.read_csv("TCGA_Barcode_breast.csv", header=0, index_col=False)  # "TCGA_Barcode" Subtype_Selected"
    y = labels["subtype_BRCA_Subtype_PAM50"].to_numpy()  # Target variable
    return X, y


def what_type(X):
    ''' The function receives a table (X) of features and checks the type of the table. '''
    if isinstance(X, pd.DataFrame):
        return 'DataFrame'
    else:
        return 'ndarray'


def clean_data(X):
    '''The function accepts a table of features and downloads from the table whose
    average is less than 20 and standard deviation is greater than 10. '''
    X = X.T
    ### Remove features which have same value for all the samples in the data.(Because of a repeated warning)
    eq = X[X.apply(lambda x: min(x) != max(x), 1)]
    indexNames = eq[(eq.mean(axis=1) <= 20) & (eq.std(axis=1) >= 10)].index
    X_clean = eq.drop(indexNames).T
    return X_clean

def norm_data(X,method='zscore'):
    if method=='zscore':
        X_norm = stats.zscore(X, axis=1)
    else:
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X)
    X_norm_df = pd.DataFrame(data=X_norm, index=list(X.index.values), columns=list(X.columns.values))
    print ("norm method: ", method)
    return X_norm_df

def clean_norm_data(X,PercentMean=25,PercentStd=20):
    """
    Delete elements that are both at PercentMean% low average and at PercentStd% low std
    :param X: DataFrame
    :param PercentMean: The low average percentage in X
    :param PercentStd: The low std percentage in X
    :return: X without elements that are both at PercentMean% low average and at PercentStd% low std
    """
    X_arr=X.to_numpy()
    nameCol_array = np.array(list(X.columns.values))
    meanX=np.mean(X_arr, axis = 0)
    sortColMean = meanX.argsort()  ### sort index from small to big by mean
    reduceMeanNum = int(round((len(meanX) * PercentMean) / 100))  ### Calculates PercentMean% of X length

    stdX = np.std(X_arr, axis=0)
    sortColStd = stdX.argsort()  ### sort index from small to big by std
    reduceStdNum = int(round((len(stdX) * PercentStd) / 100))  ### Calculates PercentStd% of X length

    mask = np.isin(sortColStd[0:reduceStdNum], sortColMean[0:reduceMeanNum]) ### Finds elements that are both at PercentMean% low average and at PercentStd% low standard deviation
    idx=np.where(mask)
    X_clean=X.drop(nameCol_array[sortColStd[idx]],axis = 1) ### Delete elements
    print ("X before clean: ", X.shape)
    print ("X_clean after clean avg & std: ", X_clean.shape)
    count = np.count_nonzero(mask)
    print("to drop avg & std: ",count)
    print ("precent to clean: avg: ", PercentMean, " std: ", PercentStd)
    return X_clean


def Fdr_anova(X, y, alpha):
    ''' Select the p-values for an estimated false discovery rate.
    This uses the Benjamini-Hochberg procedure. alpha is an upper bound on the expected false discovery rate.
    Using the ANOVA F-value for the provided sample.
    Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X. '''
    sel = SelectFdr(f_classif, alpha=alpha)
    sel.fit(X, y)

    # p(i)<=a*i/n
    # Create function that adds 100 to something
    n=np.size(X,1)
    # Create vectorized function
    vectorized_add_100 = np.vectorize(lambda i: (alpha*i)/(n+1))
    # Apply function to all elements in matrix
    alphain=vectorized_add_100(np.arange(1,n+1))

    sortColumnsP = sel.pvalues_.argsort()
    valuesortColumnsP=sel.pvalues_[sortColumnsP]
    PRDS=valuesortColumnsP<alphain  #.argsort()
    X_new_Fdr=X.iloc[:, sortColumnsP[PRDS==True]]
    print ("X before FdrAnova: ", X.shape)
    print ("X_clean after FdrAnova: ", X_new_Fdr.shape)
    print ("alpha for FdrAnova: ", alpha)
    return X_new_Fdr


def kBest_anova(X, y, k):
    ''' Select features according to the k highest scores.
    Using the ANOVA F-value for the provided sample. '''
    sel = SelectKBest(f_classif, k)
    sel.fit(X, y)
    sortColumns = sel.scores_.argsort()[::-1]
    X_new_kBest=X.iloc[:, sortColumns[:k]]
    meanarr=np.mean(X_new_kBest, axis = 0)
    stdarr=np.std(X_new_kBest, axis = 0)
    print ("X before kBest: ", X.shape)
    print ("X_clean after kBest: ", X_new_kBest.shape)
    print ("K for kBest: ", k)
    return X_new_kBest


def mutual_feature(X, y, num_neighbors, threshold):
    ''' Estimated mutual information between each feature and the target.
    It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
    The function relies on nonparametric methods based on entropy estimation from k-nearest neighbors distances. '''
    mi = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=num_neighbors)
    columns = list(X.columns.values)
    df_mi = pd.DataFrame(mi, index=columns)
    ### Drops the features that have a low dependency with Y
    indices_mutual = df_mi[(df_mi.sum(axis=1) <= threshold)].index
    df_mi = df_mi.drop(indices_mutual).T
    df_mi=df_mi.iloc[:, np.argsort(df_mi.loc[0])[::-1]]
    X_mutual=X.loc[:, df_mi.columns.values]
    print ("X before mutual: ", X.shape)
    print ("X_clean after mutual: ", X_mutual.shape)
    print ("threshold for mutual: ", threshold)
    print("median of mi: ", np.median(mi))
    return X_mutual

def Collinear_Features(fs,corr_threshold):
    ''' finds collinear features based on a specified correlation coefficient value. For each pair of correlated features,
    it identifies one of the features for removal (since we only need to remove one) '''
    fs.identify_collinear(correlation_threshold=corr_threshold)
    if fs.record_collinear.empty==True:
        return fs.data
    fs.plot_collinear()
    ### Remove the features from all methods (returns a df)
    x_Collinear_removed = fs.remove(methods=['collinear'])
    return x_Collinear_removed

def Zero_importance_Features(fs):
    ### non-deterministic
    fs.identify_zero_importance(task='classification', eval_metric='auc', n_iterations=10, early_stopping=True)
    ### list of zero importance features
    #zero_imp_features = fs.ops['zero_importance']
    ### plot the feature importances
    #fs.plot_feature_importances(threshold=0.99, plot_n=12)
    x_Zero_removed = fs.remove(methods=['zero_importance'],keep_one_hot=False)
    return x_Zero_removed

def Low_importance_Features(fs,cumulative_importance):
    ### non-deterministic
    fs.identify_zero_importance(task='classification', eval_metric='auc', n_iterations=3, early_stopping=True)
    fs.identify_low_importance(cumulative_importance=cumulative_importance)
    #low_imp_features = fs.ops['low_importance']
    x_Low_removed = fs.remove(methods=['zero_importance','low_importance'],keep_one_hot=False)
    return x_Low_removed

def Select_Features(X, y, alpha=0.05, k=0, num_neighbors=0, threshold=0, isnormal=False, isclean=False,corr_threshold=1,cumulative_importance=0):
    ''' This function gives the user the ability to choose which Feature Selection method to use.
    There are two statistical methods (1)(2) and another method of entropy(3).
    The function get X features Table, Y Label column.
    Alpha - for statistical selection (1) according to p-value.
    K- for Statistical Selection (2) selects the best K features, if K = 0 first method (1) is used instead of this method.
    num_neighbors- for entropic method (3) - gives the number of neighbors on which the selection will be based.
    threshold - for entropic method (3) - gives the boundary from which the dependence between Feature and Y is considered non-random.
    If these two values (num_neighbors & threshold)​are equal to 0 this method is not performed on X.
    isnormal - Boolean variable - True if X has already normalize, False if X has not normalize.
    isclean - Boolean variable - True if X has already clean, False if X has not clean.
    corr_threshold = Threshold for deleting high-correlated features (one of the features).
    cumulative_importance = The threshold for deleting features is of low importance.
    The function returns:
    X_new  - A table containing the features left after the statistical tests. '''

    if what_type(X) == 'ndarray': ### If X is Numpy converts it into DataFrame
        X = pd.DataFrame(data=X)

    ### If the data is already normalized do not do it again
    ### method could be 'zscore', 'minmax'
    if isnormal==False:
        X_norm=norm_data(X,method='zscore')
    else:
        X_norm=X

    ### If the data is already normalized do not clean it
    if isclean == False:
        X_clean = clean_norm_data(X_norm,PercentMean=55, PercentStd=50)
    else:
        X_clean = X_norm

    ### Selects a feature selection method
    if k != 0:
        X_new = kBest_anova(X_clean, y, k)
    else:
        X_new = Fdr_anova(X_clean, y, alpha)
    if num_neighbors != 0 and threshold != 0:
        X_new = mutual_feature(X_new, y, num_neighbors, threshold)
    if corr_threshold!=1:
        fs = FeatureSelector(data=X_new, labels=y)
        X_new = Collinear_Features(fs,corr_threshold)
    if cumulative_importance!=0:
        ### Removes low importance features
        # fs = FeatureSelector(data=X_new, labels=y)
        # X_new=Zero_importance_Features(fs)
        fs = FeatureSelector(data=X_new, labels=y)
        X_new=Low_importance_Features(fs, cumulative_importance=cumulative_importance)

    if what_type(X) == 'ndarray': ### If X is Numpy converts the results into Numpy
        X_new = X_new.to_numpy()

    return X_new

if __name__ == "__main__":
    X, y = get_X_y_data()

    X_new = Select_Features(X, y, alpha=0.05, k=50, num_neighbors=3, threshold=0.3,isnormal=False, isclean=False,corr_threshold=1,cumulative_importance=0)


