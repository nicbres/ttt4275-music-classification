import numpy as np
from statistics import mode

import source.data_handling as dh

def p_norm(p, vec):
    return np.sum(np.abs(vec)**p) ** (1/p)

#def kNN_func(
#    train_data: dh.Dataset, 
#    test_data: dh.Dataset,
#    k: int = 5,
#    p: float = 2,
#):
#    """
#    Inputs:
#        train_data: Dataset object containing the data for training
#        test_data: Dataset object containing the data for testing
#        k: number of Nearest Neighbours to be considered
#        p: order of the minkowski distance
#    """
#
#    predicted_genres = np.empty((len(test_data),), dtype=object)
#
#    for i in range(len(test_data)):
#        # Compute the p-norm to all training points
#        distances = ((train_data.x - test_data.x.iloc[i]).abs() ** p).sum(1) ** (1 / p)
#
#        # Get indices of k-smallest distances
#        smallest_indices = np.argsort(distances)[:k]
#
#        # Extract the Genres of the distance_inds from the training data_frame
#        genres_of_kNN = train_data.y.iloc[smallest_indices]
#
#        # Pick the most frequent genre
#        most_frequent_neighbour = mode(genres_of_kNN)
#
#        # Assign that genre to the track
#        predicted_genres[i] = most_frequent_neighbour
#
#    return predicted_genres


class kNN:

    def __init__(self, k:int=5, p:int=2):
        self.k = k
        self.p = p

    def fit(self, data_frame_x, data_frame_y):
        self.train_data = dh.Dataset(data_frame_x, data_frame_y)
        

    def predict(self, test_data):
        predicted_genres = np.empty((len(test_data),), dtype=object)

        for i in range(len(test_data)):
            # Compute the p-norm to all training points
            distances = ((self.train_data.x - test_data.iloc[i]).abs() ** self.p).sum(1) ** (1 / self.p)

            # Get indices of k-smallest distances
            smallest_indices = np.argsort(distances)[:self.k]

            # Extract the Genres of the distance_inds from the training data_frame
            genres_of_kNN = self.train_data.y.iloc[smallest_indices]

            # Pick the most frequent genre
            most_frequent_neighbour = mode(genres_of_kNN)

            # Assign that genre to the track
            predicted_genres[i] = most_frequent_neighbour

        return predicted_genres