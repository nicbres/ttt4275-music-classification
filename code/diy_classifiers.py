import numpy as np
import heapq
from pyexpat import features
from statistics import mode

import data_handling

def p_norm(p, vec):
    vec_powered = [vec[i]**p for i in range(len(vec))]
    return (sum(vec_powered)) ** 1/p

def kNN(
    k,
    train_data: data_handling.Dataset, 
    test_data: data_handling.Dataset,
    p=None,
    distance_metric=None
):
    """
    Inputs:     k - number of Nearest Neighbours to be considered
                data_frame - pandas dataframe of the training set
                test_frame - pandas dataframe of the test set to be classified
                feature_set - a list of the features to be used in the classifier
                p - optional arg for order of distance norm
                distance_metric - optional arg for playing with different disdence matrics
    """

    if p is None:
        p = 2
    
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

        # Find the k smallest distances
        k_smallest = heapq.nsmallest(int(k), distances)

        # Extract their indices
        kNN_inds = [distances.index(k_smallest[j]) for j in range(len(k_smallest))]

        # Extract the Genres of the distance_inds from the training data_frame
        genres_of_kNN = training_genres[kNN_inds]

        # Pick the most frequent genre
        most_frequent_neighbour = mode(genres_of_kNN)

        # Assign that genre to the track
        predicted_genres[i] = most_frequent_neighbour

    return predicted_genres


    
