from data_handling import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


X = read_genre_class_data(GENRE_CLASS_DATA_30S)
cols = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo"]


def correlations_plot(X, colset=None, colstart=None, colend=None):
    """
    Takes in dataset X (Pandas DataFrame) and a column set

    Plots a representation of the covariance matrix
    """

    if colstart is not None and colend is not None:
        # Plot using a range of columns
        df = X[:][colstart:colend]
        print(df.corr())

    elif colset is not None:
        # Plot using the column names specified in colset
        df = X[:][colset]
        print(df.corr())

    else:
        df = X[:][:]
        print(df.corr())

    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()

def principal_components_reduction(X, genres, features=None, feature_start=None, feature_end=None):
    
    if features is not None:
        X_PCA = X.loc[:,features]
    elif feature_start is not None and feature_end is not None:
        X_PCA = X.iloc[:,feature_start:feature_end]
    else:
        X_PCA = X.iloc[:,1:65]
    
    minmax_scaling = preprocessing.MinMaxScaler()
    X_minmax = minmax_scaling.fit_transform(X_PCA)

    #max_abs_scaler = preprocessing.MaxAbsScaler()
    #X_maxabs = max_abs_scaler.fit_transform(X_PCA)

    #X_std = (X_PCA - np.mean(X_PCA, axis=0))/np.std(X_PCA, axis=0)
    U, S, VT = np.linalg.svd(X_minmax, full_matrices=False)    

    Sigma = np.diag(S)
    T = U @ Sigma

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    ax.grid()

    for genre in genres:
        genre_inds = np.where(X['Genre'] == genre)
        ax.scatter(T[genre_inds, 0],T[genre_inds, 1],label=f"{genre}")
    
    ax.legend()
    plt.show()