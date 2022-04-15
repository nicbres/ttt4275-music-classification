import copy

import sklearn.preprocessing

import source.data_handling
import source.plotting
import source.neural_net
import source.mappings


ALL_FEATURES = source.mappings.MUSIC_FEATURES_ALL

def test_given_all_features_and_genres_run_mlp_and_plot_confusion_matrix():
    data_frame = source.data_handling.read_genre_class_data(
        source.data_handling.GENRE_CLASS_DATA_30S,
    )

    scaled_data_frame = copy.deepcopy(data_frame)
    scaler = sklearn.preprocessing.MaxAbsScaler()
    scaler.fit(data_frame[ALL_FEATURES])

    scaled_data_frame[ALL_FEATURES] = scaler.transform(scaled_data_frame[ALL_FEATURES])
    training_data, test_data = source.data_handling.prepare_data(
        data_frame=scaled_data_frame,
    )

    multi_layer_perceptron = source.neural_net.MLP(
        input_dimensions=len(ALL_FEATURES),
        output_dimensions=len(source.mappings.GENRES.values()),
    )

    multi_layer_perceptron.fit(
        training_data_x=training_data.x,
        training_data_y=training_data.y,
    )

    predicted_genres = multi_layer_perceptron.predict(
        test_data_x=test_data.x,
    )

    source.plotting.confusion_matrix(
        actual_genres=test_data.y,
        predicted_genres=predicted_genres,
    )


def test_given_less_features_then_run_mlp_and_confusion_matrix():
    data_frame = source.data_handling.read_genre_class_data(
        source.data_handling.GENRE_CLASS_DATA_30S,
    )

    scaled_data_frame = copy.deepcopy(data_frame)
    scaler = sklearn.preprocessing.MaxAbsScaler()
    scaler.fit(data_frame[ALL_FEATURES])

    scaled_data_frame[ALL_FEATURES] = scaler.transform(scaled_data_frame[ALL_FEATURES])

    features = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]
    training_data, test_data = source.data_handling.prepare_data(
        data_frame=scaled_data_frame,
        features=features,
    )

    multi_layer_perceptron = source.neural_net.MLP(
        input_dimensions=len(features),
        output_dimensions=len(source.mappings.GENRES.values()),
    )

    multi_layer_perceptron.fit(
        training_data_x=training_data.x,
        training_data_y=training_data.y,
    )

    predicted_genres = multi_layer_perceptron.predict(
        test_data_x=test_data.x,
    )

    source.plotting.confusion_matrix(
        actual_genres=test_data.y,
        predicted_genres=predicted_genres,
    )


def test_given_less_genres_then_run_mlp_and_confusion_matrix():
    data_frame = source.data_handling.read_genre_class_data(
        source.data_handling.GENRE_CLASS_DATA_30S,
    )

    scaled_data_frame = copy.deepcopy(data_frame)
    scaler = sklearn.preprocessing.MaxAbsScaler()
    scaler.fit(data_frame[ALL_FEATURES])

    scaled_data_frame[ALL_FEATURES] = scaler.transform(scaled_data_frame[ALL_FEATURES])

    genres = ["pop", "classical", "metal", "jazz", "disco"]
    training_data, test_data = source.data_handling.prepare_data(
        data_frame=scaled_data_frame,
        genres=genres,
    )

    multi_layer_perceptron = source.neural_net.MLP(
        input_dimensions=len(ALL_FEATURES),
        output_dimensions=len(genres),
    )

    multi_layer_perceptron.fit(
        training_data_x=training_data.x,
        training_data_y=training_data.y,
    )

    predicted_genres = multi_layer_perceptron.predict(
        test_data_x=test_data.x,
    )

    source.plotting.confusion_matrix(
        actual_genres=test_data.y,
        predicted_genres=predicted_genres,
    )
