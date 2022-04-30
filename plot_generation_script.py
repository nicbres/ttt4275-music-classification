#!/usr/bin/env python
import copy
import logging
import os
import pathlib

import sklearn.preprocessing
import sklearn.neighbors
import numpy as np

import source.data_handling
import source.plotting
import source.sklearn_reference
import source.diy_classifiers

PLOT_DIR = pathlib.Path() / "plot_dir"


def fetch_normalized_data():
    data_version = source.data_handling.GENRE_CLASS_DATA_30S
    data_frame = source.data_handling.read_genre_class_data(data_version)

    all_features = source.mappings.MUSIC_FEATURES_ALL

    #scaler = sklearn.preprocessing.MaxAbsScaler()
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(data_frame[all_features])

    data_frame[all_features] = scaler.transform(data_frame[all_features])

    return data_frame

def fetch_data():
    data_version = source.data_handling.GENRE_CLASS_DATA_30S
    data_frame = source.data_handling.read_genre_class_data(data_version)
    all_features = source.mappings.MUSIC_FEATURES_ALL

    return data_frame


def setup_output_directory(
    path: pathlib.Path = PLOT_DIR,
):
    if not os.path.exists(PLOT_DIR):
        os.path.mkdir(PLOT_DIR)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scaled_data_frame = fetch_normalized_data()
    cmap = "viridis"
    """
    # Task 1
    logging.info("===== TASK 1 =====")
    k = 5
    features_task_1 = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]

    training_data, test_data = source.data_handling.prepare_data(
        data_frame=scaled_data_frame,
        features=features_task_1,
    )

    classifier = source.diy_classifiers.kNN(k=k, p=2)
    classifier.fit(data_frame_x=training_data.x, data_frame_y=training_data.y)
    y_pred_task1 = classifier.predict(test_data.x)

    y_true_task1 = test_data.y

    source.plotting.confusion_matrix(y_true_task1, y_pred_task1, cmap=cmap)

    y_pred_task1_sklearn = source.sklearn_reference.KNN.predict(
        training_data=training_data,
        test_data=test_data,
        k=k,
        p=2,
    )

    source.plotting.confusion_matrix(y_true_task1, y_pred_task1_sklearn, cmap=cmap)

    def apply_indices_to_dataset(
        data_set,
        indices,
    ):
        data_set.x = data_set.x.iloc[indices]
        data_set.y = data_set.y.iloc[indices]
        data_set.track_ids = data_set.track_ids.iloc[indices]
        
        return data_set

    combo = ["pop", "disco"]

    train_genre_indices = np.flatnonzero(
        np.array(training_data.data_frame["Genre"] == combo[0])
        + np.array(training_data.data_frame["Genre"] == combo[1])
    )
    test_genre_indices = np.flatnonzero(
        np.array(test_data.data_frame["Genre"] == combo[0])
        + np.array(test_data.data_frame["Genre"] == combo[1])
    )
    
    result_indices = np.flatnonzero((y_pred_task1 == combo[0]) + (y_pred_task1 == combo[1])) 
    test_indices = list(filter(lambda x: x in test_genre_indices, result_indices))

    train_data_copy = copy.deepcopy(training_data)
    test_data_copy = copy.deepcopy(test_data)
    
    # Plot the training data for the two given genres and plot the data points
    # from the test data and if they were correctly / falsely classified
    source.plotting.misclassifications_scatter_plot(
        training_data=apply_indices_to_dataset(train_data_copy, train_genre_indices),
        test_data=apply_indices_to_dataset(test_data_copy, test_indices),
        predicted_genres=y_pred_task1[test_indices],
        features=features_task_1,
        genres=combo,
    )

    # Task 2
    logging.info("===== TASK 2 =====")

    features_task_2 = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]
    genres_task_2 = ["pop", "disco", "metal", "classical"]


    logging.info("===== 2a) =====")

    logging.info("==Initial Performance==")
    scaled_data_frame = fetch_data()
    features_task_2_reduced = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]
    logging.info(f"{features_task_2_reduced}")
    training_data, test_data = source.data_handling.prepare_data(
        data_frame=scaled_data_frame,
        features=features_task_2_reduced,
        genres=genres_task_2,
    )

    classifier = source.diy_classifiers.kNN(k=5, p=2)
    classifier.fit(data_frame_x=training_data.x, data_frame_y=training_data.y)
    y_pred_task1 = classifier.predict(test_data.x)

    classifier = source.sklearn_reference.KNN()
    y_pred_task1 = classifier.predict(training_data, test_data, k=5, p=2)

    y_true_task1 = test_data.y
    source.plotting.confusion_matrix(y_true_task1, y_pred_task1)

    logging.info("==Feature Distributions==")
    source.plotting.feature_distribution_histogram(
        data_frame=scaled_data_frame,
        features=features_task_2,
        genres=genres_task_2,
    )

    genres_task_2_reduced = ["pop", "disco", "metal"]
    source.plotting.feature_distribution_histogram(
        data_frame=scaled_data_frame,
        features=features_task_2,
        genres=genres_task_2_reduced,
    )

    # Task 2b
    logging.info("===== 2b) =====")
    combinations = [
        [0,1,2],
        [0,2,3],
        [0,1,3],
        [1,2,3],
    ]

    for combo in combinations:
        features = np.array(features_task_2)[combo]
        training_data, test_data = source.data_handling.prepare_data(
            data_frame=scaled_data_frame,
            features=features,
            genres=genres_task_2,
        )

        classifier = source.diy_classifiers.kNN(k=5, p=2)
        classifier.fit(data_frame_x=training_data.x, data_frame_y=training_data.y)
        y_pred_task1 = classifier.predict(test_data.x)

        #classifier = source.sklearn_reference.KNN()
        #y_pred_task1 = classifier.predict(training_data, test_data, k=5, p=2)

        y_true_task1 = test_data.y

        logging.info(f"{features}")
        source.plotting.confusion_matrix(y_true_task1, y_pred_task1)
    """
    features_task_2 = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]
    genres_task_2 = ["pop", "disco", "metal", "classical"]

    data_frame = fetch_data()
    df = data_frame.sample(frac=1)

    classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, p=2, algorithm="brute")
    training_data, test_data = source.data_handling.prepare_data(
        data_frame=df,
        features=features_task_2,
        genres=genres_task_2,
    )
    logging.info(f"Training Data X Shape: {np.shape(training_data.x)}")
    logging.info(f"Training Data Y Shape: {np.shape(training_data.y)}")

    logging.info(f"Test Data X Shape: {np.shape(test_data.x)}")
    logging.info(f"Test Data Y Shape: {np.shape(test_data.y)}")
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(training_data.x)

    training_data.x = scaler.transform(training_data.x)
    test_data.x = scaler.transform(test_data.x)

    classifier.fit(X=training_data.x, y=training_data.y)
    y_predict = classifier.predict(X=test_data.x)

    source.plotting.confusion_matrix(test_data.y, y_predict)

    with open("train_data.csv", "w") as f:
        f.write(training_data.data_frame.to_csv())

    with open("test_data.csv", "w") as f:
        f.write(test_data.data_frame.to_csv())
