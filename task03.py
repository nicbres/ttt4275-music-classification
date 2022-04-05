import source.data_handling
import source.mappings
import source.plotting
import source.diy_classifiers
from sklearn.model_selection import train_test_split

import numpy as np


data_version = source.data_handling.GENRE_CLASS_DATA_30S
data_set = source.data_handling.read_genre_class_data(data_version)
genres = 

features_task_3 = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]
remove_feature = "tempo"
features_task_3.remove(remove_feature)

training_data, _ = source.data_handling.prepare_data(
        data_frame=data_set,
        features=features_task_3,
    )

## TODO: Split training further into training and validation
data_train, data_val, y_train, y_val = train_test_split(training_data.x, training_data.y, test_size=0.2)

bruh = 1
## TODO: Perform the loop through all remaining features, predict on the validation and find which 4th feature produces the lowest error

## TODO: Predict the withold set with the winning feature