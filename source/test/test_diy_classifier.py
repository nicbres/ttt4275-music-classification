import numpy as np
import pytest

import source.data_handling
import source.mappings
import source.plotting
import source.diy_classifiers


@pytest.mark.parametrize("inputs, order, expected",
    [
        [np.array([1, 2]), 1, 3],
        [np.array([3, 4]), 2, 5],
    ],
)
def test_p_norm(inputs, order, expected):
    result = source.diy_classifiers.p_norm(p=order, vec=inputs)

    assert abs(result - expected) < 1e-9


def test_given_all_feature_and_genres_then_knn():
    data = source.data_handling.read_genre_class_data(
        file_path=source.data_handling.GENRE_CLASS_DATA_30S,
    )

    training_data, test_data = source.data_handling.prepare_data(
        data_frame=data,
    )

    y_pred_task1 = source.diy_classifiers.kNN(train_data=training_data, test_data=test_data, k=5, p=2)
    y_true_task1 = test_data.y


def test_given_less_genres_and_features_then_knn():
    data = source.data_handling.read_genre_class_data(
        file_path=source.data_handling.GENRE_CLASS_DATA_30S,
    )

    training_data, test_data = source.data_handling.prepare_data(
        data_frame=data,
        features=["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"],
        genres=["pop", "metal", "disco", "classical"],
    )

    y_pred_task1 = source.diy_classifiers.kNN(train_data=training_data, test_data=test_data, k=5, p=2)
    y_true_task1 = test_data.y

