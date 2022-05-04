#!/usr/bin/env python
import copy
import logging
import os
import pathlib

import sklearn.preprocessing
import sklearn.neighbors
import matplotlib.pyplot as plt
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
    cmap = "Blues"

    k = 5
    p = 2
    template_scaler = sklearn.preprocessing.MinMaxScaler

    logging.info("===== TASK 1 =====")
    scaler = template_scaler()
    features = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]

    training_data, test_data = source.data_handling.prepare_data(
        data_frame=source.data_handling.read_genre_class_data(
            source.data_handling.GENRE_CLASS_DATA_30S,
        ),
        features=features,
        scaler=scaler,
    )

    logging.info("DIY KNN")
    classifier = source.diy_classifiers.kNN(k=k, p=p)
    classifier.fit(data_frame_x=training_data.x, data_frame_y=training_data.y)
    y_pred_task1 = classifier.predict(test_data.x)
    y_true_task1 = test_data.y
    source.plotting.confusion_matrix(
        actual_genres=y_true_task1,
        predicted_genres=y_pred_task1,
        cmap=cmap,
        output_name="1b_diy",
    )

    logging.info("Sklearn KNN")
    classifier = sklearn.neighbors.KNeighborsClassifier(
        n_neighbors=k,
        p=p,
    )
    classifier.fit(X=training_data.x, y=training_data.y)
    y_pred_task1 = classifier.predict(test_data.x)

    source.plotting.confusion_matrix(
        actual_genres=y_true_task1,
        predicted_genres=y_pred_task1,
        cmap=cmap,
        output_name="1b_sklearn",
    )

    def apply_indices_to_dataset(
        data_set,
        indices,
    ):
        data_set.x = data_set.x.iloc[indices]
        data_set.y = data_set.y.iloc[indices]
        data_set.track_ids = data_set.track_ids.iloc[indices]
        
        return data_set

    combo = ["hiphop", "disco"]

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
        features=features,
        genres=combo,
        log_misclassified=True,
        output_name="1c",
    )

    # Task 2
    logging.info("===== TASK 2 =====")

    features = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]
    genres = ["pop", "disco", "metal", "classical"]

    logging.info("===== 2a) =====")
    logging.info("==Feature Distributions==")
    logging.info("unscaled")
    source.plotting.feature_distribution_histogram(
        data_frame=source.data_handling.read_genre_class_data(
            source.data_handling.GENRE_CLASS_DATA_30S
        ),
        features=features,
        genres=genres,
        output_name="2a_unscaled",
    )

    df = source.data_handling.read_genre_class_data(
        source.data_handling.GENRE_CLASS_DATA_30S,
    )
    training_data, test_data = source.data_handling.prepare_data(
        data_frame=df,
        features=features,
        genres=genres,
    )
    scaler = template_scaler()
    logging.info(f"{scaler}")
    scaler.fit(training_data.x[features])
    df[features] = scaler.transform(df[features])
    source.plotting.feature_distribution_histogram(
        data_frame=df,
        features=features,
        genres=genres,
        output_name=f"2a_{scaler}",
    )

    logging.info(f"{scaler} without classical")
    source.plotting.feature_distribution_histogram(
        data_frame=df,
        features=features,
        genres=["pop", "disco", "metal"],
        output_name=f"2a_{scaler}",
    )

    # Task 2b
    logging.info("===== 2b) =====")
    combinations = [
        [0,1,2],
        [0,2,3],
        [0,1,3],
        [1,2,3],
    ]

    logging.info("plot all combos of three features")
    for combo in combinations:
        new_features = np.array(features)[combo]
        training_data, test_data = source.data_handling.prepare_data(
            data_frame=source.data_handling.read_genre_class_data(
                source.data_handling.GENRE_CLASS_DATA_30S
            ),
            features=new_features,
            genres=genres,
            scaler=template_scaler(),
        )

        classifier = source.diy_classifiers.kNN(k=5, p=2)
        classifier.fit(data_frame_x=training_data.x, data_frame_y=training_data.y)
        y_pred_task1 = classifier.predict(test_data.x)
        y_true_task1 = test_data.y

        logging.info(f"{new_features}")
        features_name = ""
        for feature in new_features:
            features_name += f"_{feature}"
        source.plotting.confusion_matrix(
            actual_genres=y_true_task1,
            predicted_genres=y_pred_task1,
            output_name=f"2b_misinterpreted{features_name}",
            cmap=cmap,
        )

    features = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]
    genres = ["pop", "disco", "metal", "classical"]

    features_list = [
        ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"],
        ["spectral_rolloff_mean", "spectral_centroid_mean", "mfcc_1_mean"],
        ["spectral_rolloff_mean", "spectral_centroid_mean"],
        ["spectral_rolloff_mean"],
    ]
    for features in features_list:
        training_data, test_data = source.data_handling.prepare_data(
            data_frame=source.data_handling.read_genre_class_data(
                source.data_handling.GENRE_CLASS_DATA_30S
            ),
            features=features,
            genres=genres,
            scaler=template_scaler(),
        )

        classifier = source.diy_classifiers.kNN(k=5, p=2)
        classifier.fit(data_frame_x=training_data.x, data_frame_y=training_data.y)
        y_pred_task1 = classifier.predict(test_data.x)
        y_true_task1 = test_data.y

        logging.info(f"{features}")
        features_name = ""
        for feature in features:
            features_name += f"_{feature}"

        source.plotting.confusion_matrix(
            actual_genres=y_true_task1,
            predicted_genres=y_pred_task1,
            output_name=f"2b_{len(features)}{features_name}",
            cmap=cmap,
        )

