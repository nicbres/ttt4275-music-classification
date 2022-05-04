from cProfile import label
import logging
import math
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas
import sklearn.metrics
import numpy as np

import source.data_handling
import source.diy_classifiers
import source.descriptive_statistics
import source.sklearn_reference


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
    fig, ax = plt.subplots(1,1)

    labels = np.sort(list(set(actual_genres)))

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

    disp.plot(
        xticks_rotation=90.0,
        ax=ax,
    )

    plt.title(f"Error Rate: {error_percentage:.2f}%")

    fig.tight_layout()
    plt.show()


def error_rates_vs_params(ks, ps, training_data, test_data, diy=True):
    actual_genres = test_data.y

    error_rate_ks = np.empty((len(ks,)))
    error_rate_ps = np.empty((len(ps,)))

    for i, k in enumerate(ks):
        if diy:
            classifier = source.diy_classifiers.kNN(k=k,p=2)
        else:
            classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k,p=2)

        classifier.fit(training_data.x, training_data.y)
        predicted_genres = classifier.predict(test_data.x)

        error_percentage = source.descriptive_statistics.classifier_error_rate(predicted_genres, actual_genres)
        error_rate_ks[i] = error_percentage
        
    for j, p in enumerate(ps):
        if diy:
            classifier = source.diy_classifiers.kNN(k=5, p=p)
        else:
            classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, p=p)

        classifier.fit(training_data.x, training_data.y)
        predicted_genres = classifier.predict(test_data.x)

        error_percentage = source.descriptive_statistics.classifier_error_rate(predicted_genres, actual_genres)
        error_rate_ps[j] = error_percentage
    
    fig, axs = plt.subplots(1, 2)

    ax1 = axs[0]
    ax2 = axs[1]

    ax1.set_xlabel("k - Number of NN considered")
    ax1.set_ylabel("Error percentage [%]")

    ax2.set_xlabel("p - order of Minkowski norm")
    ax2.set_ylabel("Error percentage [%]")

    ax1.plot(ks, error_rate_ks, "o-")
    ax2.plot(ps, error_rate_ps, "o-")

    ax1.grid()
    ax2.grid()

    fig.suptitle("Error rate as a function of k and order of Minkowski norm")
    fig.tight_layout()
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
            handles, labels = axes.get_legend_handles_labels()
            plot_index += 1

    fig.legend(handles, labels)
    fig.tight_layout()

    plt.show()


def feature_distribution_histogram(
    data_frame: pandas.DataFrame,
    features: Iterable,
    genres: Iterable,
    nr_of_bins: int = 10,
):
    fig, axs = plt.subplots(2, 2)

    for index, feature in enumerate(features):
        row_index, col_index = divmod(index, 2)
        axs[row_index, col_index].set_title(feature)

        lower_bound = int(data_frame[feature].min())
        upper_bound = int(data_frame[feature].max())
        number_of_bins = 20
        bin_width = (upper_bound - lower_bound) / number_of_bins
        bins = np.arange(lower_bound, upper_bound, bin_width)

        for genre in genres:
            edge_color = matplotlib.colors.to_rgb(_COLORS[genre]) + (1.0,)
            face_color = matplotlib.colors.to_rgb(_COLORS[genre]) + (0.6,)

            data = data_frame[data_frame["Genre"] == genre][feature]


            axs[row_index, col_index].hist(
                data,
                bins,
                density=True,
                histtype='bar',
                edgecolor=edge_color,
                facecolor=face_color,
                label=genre,
            )

        axs[row_index, col_index].grid()

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels)

    fig.tight_layout()
    plt.show()

def task04_plots(cum_var, CV_vars, pc_PIs):
    nr_features = np.shape(cum_var)[0]
    x_axis = np.linspace(1, nr_features, nr_features)

    fig0 = plt.figure()
    plt.title("Explained Variance Plots")
    plt.plot(x_axis, cum_var, label="Explained Variance")
    plt.plot(x_axis, CV_vars, label="Explained Variance through Cross-Validation")
    plt.xlabel("PCA/SVD Truncation Order")
    plt.ylabel("Fraction of Explained Variance")
    plt.grid()

    fig1 = plt.figure()
    plt.title("Error Rate as a Function of Principle Component Nr")
    plt.plot(x_axis, pc_PIs, label="Cross Validation Error Rate for the PLSR-DA Classifier")
    plt.xlabel("PCA/SVD Truncation Order")
    plt.ylabel("Error Rate [%]")
    plt.grid()

    plt.legend()
    plt.show()