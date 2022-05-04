import copy
import logging

import sklearn.preprocessing
import sklearn.neighbors
import sklearn.model_selection

import source.data_handling
import source.descriptive_statistics
import source.diy_classifiers
import source.mappings
import source.plotting


def cross_validate_scaler_performance(
    features=source.mappings.MUSIC_FEATURES_ALL,
    genres=source.mappings.GENRES.values(),
    n_splits=5,
    classifier=source.diy_classifiers.kNN(),
):
    """Runs cross validation on the training data to determine performance of
    scalers.
    """
    skf = sklearn.model_selection.StratifiedKFold(n_splits=n_splits)
    scalers = [
        sklearn.preprocessing.MinMaxScaler(),
        sklearn.preprocessing.MaxAbsScaler(),
        sklearn.preprocessing.StandardScaler(),
    ]
    error_rates = [0, 0, 0]

    df = source.data_handling.read_genre_class_data(
        source.data_handling.GENRE_CLASS_DATA_30S,
    )
    full_training_data, full_test_data = source.data_handling.prepare_data(
        data_frame=df,
        features=features,
        genres=genres,
    )

    for index, scaler in enumerate(scalers):
        error_rate = 0
        for train, test in skf.split(full_training_data.x, full_training_data.y):
            df = source.data_handling.read_genre_class_data(
                source.data_handling.GENRE_CLASS_DATA_30S,
            )
            training_data, test_data = source.data_handling.prepare_data(
                data_frame=df,
                features=features,
                genres=genres,
            )

            scaler.fit(training_data.x.iloc[train])
            training_data.x[features] = scaler.transform(training_data.x[features])

            classifier.fit(training_data.x.iloc[train], training_data.y.iloc[train])
            pred_y = classifier.predict(training_data.x.iloc[test])

            error_rate += source.descriptive_statistics.classifier_error_rate(pred_y, training_data.y.iloc[test])

        df = source.data_handling.read_genre_class_data(
            source.data_handling.GENRE_CLASS_DATA_30S,
        )

        training_data, test_data = source.data_handling.prepare_data(
            data_frame=df,
            features=features,
            genres=genres,
            scaler=scaler,
        )
        
        error_rates[index] = error_rate / n_splits
        logging.info(f"{scaler} validation error rate: {error_rates[index]:.2f}")

        classifier.fit(training_data.x, training_data.y)
        pred_y = classifier.predict(test_data.x)
        error_rate = source.descriptive_statistics.classifier_error_rate(pred_y, test_data.y)
        logging.info(f"{scaler} test error rate: {error_rate:.2f}")

    return scalers, error_rates


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    features = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]
    logging.info(f"Features: {features}")
    cross_validate_scaler_performance(
    #    features=features,
    )
