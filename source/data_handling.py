import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import librosa
import numpy as np
import pandas as pd

import source.mappings

_file_path = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = Path(_file_path) / ".." / "resources" / "data"
GENRE_CLASS_DATA_5S = DATA_DIR / "GenreClassData_5s.txt"
GENRE_CLASS_DATA_10S = DATA_DIR / "GenreClassData_10s.txt"
GENRE_CLASS_DATA_30S = DATA_DIR / "GenreClassData_30s.txt"


@dataclass
class Dataset:
    x: pd.DataFrame
    y: pd.DataFrame


def read_genre_class_data(
    file_path: Path,
) -> pd.DataFrame:
    if file_path.is_file() is False:
        raise ValueError(f"Given file path is not a file ({file_path})")
    return pd.read_table(file_path)


def reduce_genres(
    data_frame: pd.DataFrame,
    genres: Iterable,
) -> pd.DataFrame:
    query = ""
    for index, genre in enumerate(genres):
        if index != 0:
            query += " | "
        query += f"Genre == '{genre}'"

    return data_frame.query(query)


def reduce_features(
    data_frame: pd.DataFrame,
    features: Iterable,
) -> pd.DataFrame:
    return data_frame.loc[:, features]


def prepare_data(
    data_frame: pd.DataFrame,
    genres: Optional[Iterable] = None,
    features: Optional[Iterable] = None,
) -> Tuple[Dataset, Dataset]:
    """Runs the scikit learn KNN classifier.

    Uses the specified genres and features for Classification:

    Args:
        data_frame: The pandas data frame containing the feature vector values.
        genres: An iterable of genres available in the data frame.
        features: An iterable of feautres available in the data frame.
    """

    if genres is not None:
        available_genres = set(data_frame["Genre"])
        if any([genre not in available_genres for genre in genres]):
            raise ValueError("One of the specified genres is invalid")
    else:
        genres = source.mappings.GENRES.values()

    if features is not None:
        if any([feature not in data_frame.keys() for feature in features]):
            raise ValueError("One of the specified features is invalid")
    else:
        features = source.mappings.MUSIC_FEATURES_ALL

    training_df = data_frame[data_frame["Type"] == "Train"]
    test_df = data_frame[data_frame["Type"] == "Test"]

    training_data = Dataset(
        x=reduce_features(reduce_genres(training_df, genres), features),
        y=reduce_genres(training_df, genres)["Genre"],
    )

    test_data = Dataset(
        x=reduce_features(reduce_genres(test_df, genres), features),
        y=reduce_genres(test_df, genres)["Genre"],
    )

    return training_data, test_data


def read_wav_file(
    file_path: Path,
    duration: int = 30,  # [s]
):
    audio_data, sample_rate = librosa.load(file_path, duration=duration)
    return audio_data, sample_rate


def split_audio_data_into_segments(
    audio_data: np.ndarray,
    sample_rate: float,
    segment_length: int,  # [s]
):
    samples_per_segment = sample_rate * segment_length
    number_of_segments = len(audio_data) // samples_per_segment

    audio_data = audio_data[: number_of_segments * samples_per_segment]

    return np.reshape(audio_data, (number_of_segments, samples_per_segment))
