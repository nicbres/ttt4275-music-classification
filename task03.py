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
    perform_ind = source.descriptive_statistics.classifier_error_rate(y_hat, y_true)     
    return perform_ind

def cross_validate(training_x, training_y, nr_segments=5):

    genres = list(set(training_y))
    #x_split = np.array_split(training_x, nr_segments)
    y_split = np.array_split(training_y, nr_segments)
    
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

        # Make into correct object for kNN classifier
        train_data_n = source.data_handling.Dataset(x=train_data_x, y=train_data_y)
        val_data_n = source.data_handling.Dataset(x=val_data_x, y=val_data_y)
        
        # Estimate y
        y_hat_n = source.diy_classifiers.kNN(k=5, p=2, train_data=train_data_n, test_data=val_data_n)

        # Performance index computation
        PI_n = PI(val_data_y, y_hat_n)
        PI_segments[i] = PI_n
    
    return np.average(PI_segments)
        

def model_structure_selection(training_set, fast_features, add_features):
    
    add_features_PIs = np.empty((len(add_features), ))

    for i, add_feature in enumerate(add_features):

        features_n = fast_features.copy()
        features_n.append(add_feature)

        feature_PI = cross_validate(
            training_x=training_set.x[features_n],
            training_y=training_set.y,
        )
        add_features_PIs[i] = feature_PI
        
    return add_features_PIs



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