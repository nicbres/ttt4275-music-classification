import numpy as np
from statistics import mode

import source.data_handling as dh

def p_norm(p, vec):
    return np.sum(vec**p) ** (1/p)

def kNN(
    train_data: dh.Dataset, 
    test_data: dh.Dataset,
    k: int = 5,
    p: float = 2,
):
    """
    Inputs:     k - number of Nearest Neighbours to be considered
                data_frame - pandas dataframe of the training set
                test_frame - pandas dataframe of the test set to be classified
                feature_set - a list of the features to be used in the classifier
                p - optional arg for order of distance norm
                distance_metric - optional arg for playing with different disdence matrics
    """

    # TODO: play around with other metrics

    N_train = np.shape(train_data.x)[0]
    N_test = np.shape(test_data.x)[0]

    training_table = train_data.x
    training_genres = train_data.y

    test_table = test_data.x
    predicted_genres = np.empty((N_test, ), dtype=object)

    for i in range(N_test):
        # Compute the euclidean distances to all training points
        distances = [p_norm(p,test_table.iloc[i]-training_table.iloc[j]) for j in range(N_train)]

        # Get indices of k-smallest distances
        smallest_indices = np.argsort(distances)[:k]

        # Extract the Genres of the distance_inds from the training data_frame
        genres_of_kNN = training_genres.iloc[smallest_indices]

        # Pick the most frequent genre
        most_frequent_neighbour = mode(genres_of_kNN)

        # Assign that genre to the track
        predicted_genres[i] = most_frequent_neighbour

    return predicted_genres
