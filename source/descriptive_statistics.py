import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

import source.mappings
import source.diy_classifiers


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


def perform_PCA(
    X,
    genres=source.mappings.GENRES.values(),
    features=source.mappings.MUSIC_FEATURES_ALL,
    do_validation=True
):
    """
    Takes in a data frame, genres and features and performs PCA on the dataset.
    Returns 
    """
    X_PCA = X.loc[:,features]
    n_samples = X_PCA.shape[0]
    n_features = X_PCA.shape[1]

    U, singular_values, VT = np.linalg.svd(X_PCA, full_matrices=False)    

    Sigma_matrix = np.diag(singular_values)
    Scores = U @ Sigma_matrix

    explained_var = (Sigma_matrix**2) / (n_samples - 1)
    total_var = np.sum(explained_var)
    explained_var_ratios = explained_var / total_var

    variances = np.diag(explained_var_ratios)
    cum_var = np.cumsum(variances)

    return Scores, VT.T, singular_values, cum_var 


def principal_components_reduction_plot(
    X,
    genres=source.mappings.GENRES.values(),
    features=source.mappings.MUSIC_FEATURES_ALL,
):
    T = principal_components_reduction(
        X=X,
        genres=genres,
        features=features,
    )
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

def classifier_error_rate(predicted_genres, actual_genres):
    errors = actual_genres != predicted_genres
    error_percentage = sum(errors) / len(actual_genres) * 100
    return error_percentage


def cross_validate(training_x, training_y, nr_segments=5, classifier=source.diy_classifiers.kNN):

    genres = list(set(training_y))
    
    # genres split is a list containing 10 single_genre_split lists
    # single_genre_split is a list containing 5 lists of data_frames
    genres_split_x = []
    genres_split_y = []

    for genre in genres:
        single_genre_inds = np.where(training_y == genre)
        single_genre_split_x = np.array_split(
            training_x.iloc[single_genre_inds],
            nr_segments,
        )

        single_genre_split_y = np.array_split(
            training_y.iloc[single_genre_inds],
            nr_segments,
        )

        genres_split_x.append(single_genre_split_x)
        genres_split_y.append(single_genre_split_y)

    PI_segments = np.empty((nr_segments, ))

    for i in range(nr_segments):
        
        # 
        all_genres_x_val = []
        all_genres_y_val = []

        all_genres_x_train = []
        all_genres_y_train = []

        for j in range(len(genres)):
            single_genre_split_x = genres_split_x[j].copy()
            single_genre_split_y = genres_split_y[j].copy()

            single_genre_x_val = single_genre_split_x[i].copy()
            single_genre_y_val = single_genre_split_y[i].copy()

            all_genres_x_val.append(single_genre_x_val)
            all_genres_y_val.append(single_genre_y_val)

            single_genre_split_x.pop(i)
            single_genre_split_y.pop(i)
            
            for k in range(len(single_genre_split_y)):
                all_genres_x_train.append(single_genre_split_x[k])
                all_genres_y_train.append(single_genre_split_y[k])

        val_data_x = pd.concat(all_genres_x_val)
        val_data_y = pd.concat(all_genres_y_val)

        train_data_x = pd.concat(all_genres_x_train)
        train_data_y = pd.concat(all_genres_y_train)
        
        # fit classifier on the training set
        classifier.fit(train_data_x, train_data_y)

        # Estimate y
        y_hat_n = classifier.predict(val_data_x)

        # Performance index computation
        PI_n = classifier_error_rate(val_data_y, y_hat_n)
        PI_segments[i] = PI_n
    
    return np.average(PI_segments)


def cross_validate_pca(training_x, training_y, nr_segments=5, classifier=source.diy_classifiers.kNN):

    genres = list(set(training_y))
    
    # genres split is a list containing 10 single_genre_split lists
    # single_genre_split is a list containing 5 lists of data_frames
    genres_split_x = []
    genres_split_y = []

    for genre in genres:
        single_genre_inds = np.where(training_y == genre)
        single_genre_split_x = np.array_split(
            training_x.iloc[single_genre_inds],
            nr_segments,
        )

        single_genre_split_y = np.array_split(
            training_y.iloc[single_genre_inds],
            nr_segments,
        )

        genres_split_x.append(single_genre_split_x)
        genres_split_y.append(single_genre_split_y)

    vars_segments = []
    for i in range(nr_segments):
        
        # 
        all_genres_x_val = []
        all_genres_y_val = []

        all_genres_x_train = []
        all_genres_y_train = []

        for j in range(len(genres)):
            single_genre_split_x = genres_split_x[j].copy()
            single_genre_split_y = genres_split_y[j].copy()

            single_genre_x_val = single_genre_split_x[i].copy()
            single_genre_y_val = single_genre_split_y[i].copy()

            all_genres_x_val.append(single_genre_x_val)
            all_genres_y_val.append(single_genre_y_val)

            single_genre_split_x.pop(i)
            single_genre_split_y.pop(i)
            
            for k in range(len(single_genre_split_y)):
                all_genres_x_train.append(single_genre_split_x[k])
                all_genres_y_train.append(single_genre_split_y[k])

        val_data_x = pd.concat(all_genres_x_val)
        val_data_y = pd.concat(all_genres_y_val)

        train_data_x = pd.concat(all_genres_x_train)
        train_data_y = pd.concat(all_genres_y_train)
        
        # fit classifier on the training set

        n_samples = train_data_x.shape[0]
        U, singular_values, VT = np.linalg.svd(train_data_x, full_matrices=False)    

        Sigma_matrix = np.diag(singular_values)

        explained_var = (Sigma_matrix**2) / (n_samples - 1)
        total_var = np.sum(explained_var)

        explained_var_ratios = explained_var / total_var

        variances = np.diag(explained_var_ratios)
        cum_var = np.cumsum(variances)

        # Performance index computation
        vars_segments.append(cum_var)

    return np.average(vars_segments, 0)
