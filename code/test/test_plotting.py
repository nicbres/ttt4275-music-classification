import data_handling
import plotting
import sklearn_knn


def test_confusion_matrix_plot():
    data = data_handling.read_genre_class_data(
        file_path=data_handling.GENRE_CLASS_DATA_30S,
    )

    training_data, test_data = data_handling.prepare_data(data)

    predicted_genres = sklearn_knn.predict(training_data, test_data)

    plotting.confusion_matrix(
        actual_genres=test_data.y,
        predicted_genres=predicted_genres,
    )

