import source.data_handling
import source.mappings
import source.plotting
import source.diy_classifiers
import source.descriptive_statistics
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

def PI(y_true, y_hat, which_pi=None):
    # TODO: implement different PIs
    source.descriptive_statistics.classifier_error_rate(y_hat, y_true)
    return PI

def cross_validate(training_x, training_y, nr_segments=5):


    x_split = np.array_split(training_x, nr_segments)
    y_split = np.array_split(training_y, nr_segments)

    PI_segments = np.empty((nr_segments, ))

    for i in range(nr_segments):

        # Extract validation set
        x_val_n = x_split[i].copy()
        y_val_n = y_split[i].copy()

        # Create a copy of training set
        x_train_n = x_split.copy()
        y_train_n = y_split.copy()

        # Remove validation set
        x_train_n.pop(i)
        y_train_n.pop(i)

        # Remaining become a big dataframe
        x_train_merged = pd.concat(x_train_n)
        y_train_merged = pd.concat(y_train_n)

        # Make into correct object for kNN classifier
        train_data_n = source.data_handling.Dataset(x=x_train_merged, y=y_train_merged)
        val_data_n = source.data_handling.Dataset(x=x_val_n, y=y_val_n)
        
        # Estimate y
        y_hat_n = source.diy_classifiers.kNN(k=5, train_data=train_data_n, test_data=val_data_n, p=2)

        # Performance index computation
        PI_n = PI(y_val_n, y_hat_n)
        PI_segments[i] = PI_n
    
    return np.average(PI_segments)
        

def model_structure_selection(training_set, fast_features, add_features, genres):
    
    add_features_PIs = np.empty((len(add_features), ))

    for i,add_feature in add_features:

        features_n = fast_features.copy()
        features_n.append(add_features[add_feature])

        training_n = training_set[features_n]
        feature_PI = cross_validate(training_n.x, training_n.y)
        add_features_PIs[i] = feature_PI
        
    return add_features_PIs



data_version = source.data_handling.GENRE_CLASS_DATA_30S
data_set = source.data_handling.read_genre_class_data(data_version)

features_task_3 = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]
remove_feature = "tempo"
features_task_3.remove(remove_feature)

# Extract the dataset with full features here, will chose specific in model order selection function
training_data, _ = source.data_handling.prepare_data(
        data_frame=data_set,
        features=features_task_3,
    )

## TODO: Split training further into training and validation
data_train, data_val, y_train, y_val = train_test_split(training_data.x, training_data.y, test_size=0.2)

bruh = 1
## TODO: Perform the loop through all remaining features, predict on the validation and find which 4th feature produces the lowest error

## TODO: Predict the withold set with the winning feature