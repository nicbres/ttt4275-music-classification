from pyexpat import features
import numpy as np
import heapq


def diy_kNN(k,
            data_frame, 
            test_frame,
            feature_set, 
            distance_metric=None
            ):
    """
    Inputs:     k - number of Nearest Neighbours to be considered
                data_frame - pandas dataframe of the training set
                test_frame - pandas dataframe of the test set to be classified
                feature_set - a list of the features to be used in the classifier
                distance_metric - optional arg for playing with different disdence matrics
    """

    if distance_metric is None:
        distance_metric = "euclidean"
    else:
        bruh = 1 # TODO: play around with other metrics
    
    #class_LUT = data_frame[["Track ID", "Genre"]]

    training_columns = feature_set.insert(0, "Genre")
    training_table = data_frame[training_columns]

    N_train = np.shape(data_frame)[0]
    N_test = np.shape(test_frame)[0]
    
    test_table = test_frame[feature_set]

    for i in range(N_test):
        
        # Compute the euclidean distances to all training points
        distances = [np.linalg.norm(test_table[i]-training_table[j]) for j in range(N_train)]

        # Find the k smallest distances
        k_smallest = heapq.nsmallest(int(k), distances)

        # Extract their indices
        kNN_inds = [distances.index(k_smallest[j] for j in range(len(k_smallest)))]

        # Extract the Genres of the distance_inds from the training data_frame
        genres_of_kNN = 1 # TODO
        # Pick the most frequent genre

        # 


        



    