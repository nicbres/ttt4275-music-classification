import sklearn
import source.data_handling
import source.mappings
import source.plotting
import source.diy_classifiers
import source.descriptive_statistics
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


def model_structure_selection(training_set, fast_features, add_features):
    
    add_features_PIs = np.empty((len(add_features), ))
    kNN_classifier = source.diy_classifiers.kNN(k=5, p=2)
    #kNN_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, p=2)

    for i, add_feature in enumerate(add_features):

        features_n = fast_features.copy()
        features_n.append(add_feature)

        feature_PI = source.descriptive_statistics.cross_validate(
            training_x=training_set.x[features_n],
            training_y=training_set.y,
            classifier=kNN_classifier,
        )
        add_features_PIs[i] = feature_PI
        
    return add_features_PIs


if __name__ == "__main__":
    data_version = source.data_handling.GENRE_CLASS_DATA_30S
    data_set = source.data_handling.read_genre_class_data(data_version)

    features_task_3 = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]
    remove_feature_ind = 1
    features_task_3.pop(remove_feature_ind)

    add_features = source.mappings.MUSIC_FEATURES_ALL.copy()
    for feature in features_task_3:
        add_features.remove(feature)    

    # Extract the dataset with full features here, will chose specific in model order selection function
    training_data, _ = source.data_handling.prepare_data(
            data_frame=data_set,
            features=source.mappings.MUSIC_FEATURES_ALL,
        )


    add_features_PIs = model_structure_selection(training_set=training_data, fast_features=features_task_3, add_features=add_features)

    best_ind = np.argmin(add_features_PIs)

    print(add_features_PIs)
    print(f"The best extra feature to add is: {add_features[best_ind]} with an Error Rate of: {add_features_PIs[best_ind]}")
