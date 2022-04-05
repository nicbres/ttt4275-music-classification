import logging

import pytest

import source.data_handling
import source.diy_classifiers
import source.mappings
import source.plotting
import source.sklearn_knn


@pytest.mark.parametrize("features",
    [
        source.mappings.MUSIC_FEATURES_ALL,
        source.mappings.get_features_from_indices([11, 42, 7, 41]),
    ],
)
def test_confusion_matrix_plot_sklearn_knn(features):
    data = source.data_handling.read_genre_class_data(
        file_path=source.data_handling.GENRE_CLASS_DATA_30S,
    )

    training_data, test_data = source.data_handling.prepare_data(
        data_frame=data,
        features=features,
    )

    predicted_genres = source.sklearn_knn.predict(training_data, test_data)

    source.plotting.confusion_matrix(
        actual_genres=test_data.y,
        predicted_genres=predicted_genres,
    )


@pytest.mark.parametrize("features",
    [
        source.mappings.MUSIC_FEATURES_ALL,
        source.mappings.get_features_from_indices([11, 42, 7, 41]),
    ],
)
def test_confusion_matrix_plot_diy_knn(features):
    data = source.data_handling.read_genre_class_data(
        file_path=source.data_handling.GENRE_CLASS_DATA_30S,
    )

    training_data, test_data = source.data_handling.prepare_data(
        data_frame=data,
        features=features,
    )

    predicted_genres = source.diy_classifiers.kNN(
        k=5,
        train_data=training_data, 
        test_data=test_data,
    )

    source.plotting.confusion_matrix(
        actual_genres=test_data.y,
        predicted_genres=predicted_genres,
    )


@pytest.mark.parametrize("features",
    [
        source.mappings.get_features_from_indices([11, 42]),
        source.mappings.get_features_from_indices([11, 42, 7]),
        source.mappings.get_features_from_indices([11, 42, 7, 41]),
        source.mappings.get_features_from_indices([11, 42, 7, 41, 43]),
    ],
)
def test_2d_scatter_plot(features):
    data = source.data_handling.read_genre_class_data(
        file_path=source.data_handling.GENRE_CLASS_DATA_30S,
    )

    source.plotting.scatter_plot(
        data_frame=data,
        features=features,
        genres=source.mappings.GENRES.values(),
    )


@pytest.mark.parametrize("features",
    [
        source.mappings.get_features_from_indices([11, 42]),
        source.mappings.get_features_from_indices([11, 42, 7]),
        source.mappings.get_features_from_indices([11, 42, 7, 41]),
        source.mappings.get_features_from_indices([11, 42, 7, 41, 43]),
    ],
)
def test_misclassifications_scatter_plot(features, caplog):
    caplog.set_level(logging.INFO)

    data = source.data_handling.read_genre_class_data(
        file_path=source.data_handling.GENRE_CLASS_DATA_30S,
    )

    training_data, test_data = source.data_handling.prepare_data(
        data_frame=data,
        features=features,
    )

    predicted_genres = source.sklearn_knn.predict(training_data, test_data)

    source.plotting.misclassifications_scatter_plot(
        training_data=training_data,
        test_data=test_data,
        predicted_genres=predicted_genres,
        features=features,
        genres=["pop", "classical"],
        log_misclassified=True,
    )


@pytest.mark.parametrize("features",
    [
        source.mappings.get_features_from_indices([11, 42, 7, 41]),
    ],
)
def test_feature_distribution_histogram(features):
    data = source.data_handling.read_genre_class_data(
        file_path=source.data_handling.GENRE_CLASS_DATA_30S,
    )

    source.plotting.feature_distribution_histogram(
        data_frame=data,
        features=features,
        genres=["pop", "disco", "metal", "classical"],
    )
