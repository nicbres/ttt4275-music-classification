from data_handling import *
import numpy as np
import matplotlib.pyplot as plt


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

correlations_plot(X, colset=cols)
