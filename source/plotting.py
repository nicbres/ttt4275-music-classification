from cProfile import label
import logging
import math
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas
import sklearn.metrics
import numpy as np

import source.data_handling
import source.diy_classifiers
import source.descriptive_statistics
import source.sklearn_knn


_COLORS = {
    "pop": "tab:blue",
    "metal": "tab:orange",
    "disco": "tab:purple",
    "blues": "tab:brown",
    "reggae": "tab:pink",
    "classical": "tab:grey",
    "rock": "tab:cyan",
    "hiphop": "gold",
    "country": "darkblue",
    "jazz": "mediumaquamarine",
}

_MARKERS = {
    "pop": "o",
    "metal": "v",
    "disco": "^",
    "blues": "<",
    "reggae": ">",
    "classical": "p",
    "rock": "*",
    "hiphop": "X",
    "country": "d",
    "jazz": "s",
}


def single_scatter_plot(
    data_frame: pandas.DataFrame,
    features: Iterable,
    genres: Iterable,
    use_color_dict: bool,
    ax: plt.axis,
    marker: str = "o",
):
    """Generates a scatter plot from the given genres and features.

    Args:
        data_frame: The pandas DataFrame that corresponds to the music features.
        features: Features to use in plot. MUST be 2.
        genres: An iterable of genres to include in the scatter plot.
    """
    if len(features) != 2:
        raise ValueError(f"Must give exactly 2 features, but {len(features)} given.")

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.grid()

    for genre in genres:
        genre_specific_data = data_frame[data_frame["Genre"] == genre]
        x = genre_specific_data[features[0]]
        y = genre_specific_data[features[1]]

        color = _COLORS[genre] if use_color_dict else None
        ax.scatter(x, y, color=color, marker=marker, label=f"{genre}")


def scatter_plot(
    data_frame: pandas.DataFrame,
    features: Iterable,
    genres: Iterable,
    use_color_dict: bool = False,
):
    """Generates a scatter plot from the given genres and features."""

    number_of_features = len(features)
    number_of_subplots = np.sum(np.arange(number_of_features))

    number_of_subplots_cols = int(math.sqrt(number_of_subplots))
    while number_of_subplots_cols > 0:
        number_of_subplots_rows, modulo = divmod(number_of_subplots, number_of_subplots_cols)
        if modulo == 0:
            break
        number_of_subplots_cols -= 1

    logging.debug(f"Plot Shape: {number_of_subplots_rows}, {number_of_subplots_cols}")
    fig, axs = plt.subplots(number_of_subplots_rows, number_of_subplots_cols)
    plt.set_loglevel("info")

    logging.debug("Generating multiple scatter plots")
    plot_index = 0
    for i in range(number_of_features - 1):
        for j in range(i + 1, number_of_features):
            subplot_index_row, subplot_index_col = divmod(
                plot_index,
                number_of_subplots_cols,
            )
            if number_of_subplots == 1:
                axes = axs
            elif number_of_subplots_cols == 1:
                axes = axs[subplot_index_row]
            else:
                axes = axs[subplot_index_row, subplot_index_col]

            single_scatter_plot(
                data_frame=data_frame,
                features=[features[i], features[j]],
                genres=genres,
                ax=axes,
                use_color_dict=use_color_dict,
            )
            axes.legend()
            plot_index += 1

    plt.show()


def threed_scatter(
    data_frame: pandas.DataFrame,
    features: Iterable,
    genres: Iterable,
):
    """
    Generates a 3D scatter plot from the given genres and 3 features
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    ax.grid()

    for genre in genres:
        genre_specific_data = data_frame[data_frame["Genre"] == genre]
        x = genre_specific_data[features[0]]
        y = genre_specific_data[features[1]]
        z = genre_specific_data[features[2]]

        ax.scatter(x, y, z, label=f"{genre}")

    ax.legend()
    plt.show()


def confusion_matrix(
    actual_genres: Iterable,
    predicted_genres: Iterable,
):
    labels = list(set(actual_genres))

    error_percentage = source.descriptive_statistics.classifier_error_rate(predicted_genres, actual_genres)

    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_true=actual_genres,
        y_pred=predicted_genres,
        labels=labels,
    )

    disp = sklearn.metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=labels,
    )

    disp.plot()
    plt.title(f"Error Rate: {error_percentage:.2f}%")
    plt.show()


def error_rates_vs_params(ks, ps, training_data, test_data):

    actual_genres = test_data.y

    error_rate_ks = np.empty((len(ks,)))
    error_rate_ps = np.empty((len(ps,)))

    for i in range(len(ks)):
        predicted_genres = source.sklearn_knn.predict(training_data, test_data,  k=ks[i], p=2)
        error_percentage = source.descriptive_statistics.classifier_error_rate(predicted_genres, actual_genres)
        error_rate_ks[i] = error_percentage
        
    for j in range(len(ps)):
        predicted_genres = source.sklearn_knn.predict(training_data, test_data,  k=5, p=ps[j])
        error_percentage = source.descriptive_statistics.classifier_error_rate(predicted_genres, actual_genres)
        error_rate_ps[j] = error_percentage
    
    fig, axs = plt.subplots(1, 2)

    ax1 = axs[0]
    ax2 = axs[1]

    ax1.set_xlabel("k - Number of NN considered")
    ax1.set_ylabel("Error percentage [%]")

    ax2.set_xlabel("p - order of Minkowski norm")
    ax2.set_ylabel("Error percentage [%]")

    ax1.plot(ks, error_rate_ks)
    ax2.plot(ps, error_rate_ps)

    ax1.grid()
    ax2.grid()

    plt.title("Error rate as a function of k and order of Minkowski norm")
    plt.show()
    

def _classified_data_scatter_plot(
    test_data: source.data_handling.Dataset,
    predicted_genres: Iterable,
    features: Iterable,
    genres: Iterable,
    ax: plt.axis,
    log_misclassified: bool,
):
    class_is_correct = test_data.y == predicted_genres

    for genre in genres:
        indices = np.flatnonzero(test_data.data_frame["Genre"] == genre)

        classified_correctly = np.flatnonzero(class_is_correct.iloc[indices] == True)
        classified_falsely = np.flatnonzero(class_is_correct.iloc[indices] == False)

        if log_misclassified:
            lookup_table = pandas.read_table(source.mappings.MAP_DATA_FILE)
            for misclassified_index in classified_falsely:
                entry = test_data.data_frame.iloc[indices[misclassified_index]]
                actual_genre = entry["Genre"]
                track_id = entry["Track ID"]
                file_name = lookup_table[lookup_table["Track ID"] == track_id]["FileName"].to_string(index=False)
                predicted_genre = predicted_genres[indices[misclassified_index]]
                logging.info(f"ID: {track_id} ({file_name}) misclassified as {predicted_genre}")

        x = test_data.x[features[0]]
        y = test_data.x[features[1]]

        ax.scatter(
            x.iloc[indices[classified_correctly]],
            y.iloc[indices[classified_correctly]],
            color="g",
            marker=_MARKERS[genre],
            label=f"Correct {genre}",
        )
        ax.scatter(
            x.iloc[indices[classified_falsely]],
            y.iloc[indices[classified_falsely]],
            color="r",
            marker=_MARKERS[genre],
            label=f"Misclassified {genre}",
        )


def misclassifications_scatter_plot(
    training_data: source.data_handling.Dataset,
    test_data: source.data_handling.Dataset,
    predicted_genres: Iterable,
    features:Iterable,
    genres: Iterable,
    use_color_dict: bool = True,
    log_misclassified: bool = False,
):
    """Generates a scatter plot from the given genres and features."""
    number_of_features = len(features)
    number_of_subplots = np.sum(np.arange(number_of_features))
     
    number_of_subplots_cols = int(math.sqrt(number_of_subplots))
    while number_of_subplots_cols > 0:
        number_of_subplots_rows, modulo = divmod(number_of_subplots, number_of_subplots_cols)
        if modulo == 0:
            break
        number_of_subplots_cols -= 1

    logging.debug(f"Plot Shape: {number_of_subplots_rows}, {number_of_subplots_cols}")
    fig, axs = plt.subplots(number_of_subplots_rows, number_of_subplots_cols)
    plt.set_loglevel("info")

    logging.debug("Generating multiple scatter plots")
    plot_index = 0
    for i in range(number_of_features - 1):
        for j in range(i + 1, number_of_features):
            subplot_index_row, subplot_index_col = divmod(
                plot_index,
                number_of_subplots_cols,
            )
            if number_of_subplots == 1:
                axes = axs
            elif number_of_subplots_cols == 1:
                axes = axs[subplot_index_row]
            else:
                axes = axs[subplot_index_row, subplot_index_col]

            single_scatter_plot(
                data_frame=training_data.data_frame,
                features=[features[i], features[j]],
                genres=genres,
                use_color_dict=use_color_dict,
                ax=axes,
                marker="1",
            )
            _classified_data_scatter_plot(
                test_data=test_data,
                predicted_genres=predicted_genres,
                features=[features[i], features[j]],
                genres=genres,
                ax=axes,
                log_misclassified=log_misclassified if plot_index == 0 else False,
            )
            axes.legend()
            plot_index += 1

    plt.show()

