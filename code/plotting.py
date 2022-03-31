import logging
import math
from typing import Iterable

import matplotlib.pyplot as plt
import pandas


def single_scatter_plot(
    data_frame: pandas.DataFrame,
    features: Iterable,
    genres: Iterable,
    ax,
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
        genre_specific_data = data_frame[data_frame['Genre'] == genre]
        x = genre_specific_data[features[0]]
        y = genre_specific_data[features[1]]

        ax.scatter(x,y, label=f"{genre}")

    ax.legend()


def scatter_plot(
    data_frame: pandas.DataFrame,
    features: Iterable,
    genres: Iterable,
):
    """Generates a scatter plot from the given genres and features."""
    
    number_of_features = len(features)
    number_of_subplots = math.factorial(number_of_features - 1)

    number_of_subplots_cols = int(math.sqrt(number_of_subplots))
    number_of_subplots_rows = number_of_subplots // number_of_subplots_cols
    
    logging.debug(f"Plot Shape: {number_of_subplots_rows}, {number_of_subplots_cols}")
    fig, axs = plt.subplots(number_of_subplots_rows, number_of_subplots_cols)
    plt.set_loglevel("info")

    if number_of_subplots_rows > 1:
        logging.debug("Generating multiple scatter plots")
        plot_index = 0
        for i in range(number_of_features - 1):
            for j in range(i + 1, number_of_features):
                subplot_index_row, subplot_index_col = divmod(
                    plot_index, number_of_subplots_cols,
                )
                single_scatter_plot(
                    data_frame=data_frame,
                    features=[features[i], features[j]],
                    genres=genres,
                    ax=axs[subplot_index_row, subplot_index_col],
                )
                plot_index += 1
    else:
        logging.debug("Generating single scatter plots")
        single_scatter_plot(
            data_frame=data_frame,
            features=[features[0], features[1]],
            genres=genres,
            ax=axs,
        )

    plt.show()

