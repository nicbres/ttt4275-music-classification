import pytest

import data_handling
import diy_classifiers
import mappings
import plotting
import sklearn_knn


@pytest.mark.parametrize("features",
    [
        mappings.MUSIC_FEATURES_ALL,
        mappings.get_features_from_indices([11, 42, 7, 41]),
    ],
)
def test_confusion_matrix_plot_sklearn_knn(features):
    data = data_handling.read_genre_class_data(
        file_path=data_handling.GENRE_CLASS_DATA_30S,
    )

    training_data, test_data = data_handling.prepare_data(
        data_frame=data,
        features=features,
    )

    predicted_genres = sklearn_knn.predict(training_data, test_data)

    plotting.confusion_matrix(
        actual_genres=test_data.y,
        predicted_genres=predicted_genres,
    )


@pytest.mark.parametrize("features",
    [
        mappings.MUSIC_FEATURES_ALL,
        mappings.get_features_from_indices([11, 42, 7, 41]),
    ],
)
def test_confusion_matrix_plot_diy_knn(features):
    data = data_handling.read_genre_class_data(
        file_path=data_handling.GENRE_CLASS_DATA_30S,
    )

    training_data, test_data = data_handling.prepare_data(
        data_frame=data,
        features=features,
    )

    predicted_genres = diy_classifiers.kNN(
        k=5,
        train_data=training_data, 
        test_data=test_data,
    )

    plotting.confusion_matrix(
        actual_genres=test_data.y,
        predicted_genres=predicted_genres,
    )
