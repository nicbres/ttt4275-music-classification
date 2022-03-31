import os
from pathlib import Path

import librosa
import numpy as np
import pandas


DATA_DIR = Path() / '..' / 'resources' / 'data'
GENRE_CLASS_DATA_5S = DATA_DIR / 'GenreClassData_5s.txt'
GENRE_CLASS_DATA_10S = DATA_DIR / 'GenreClassData_10s.txt'
GENRE_CLASS_DATA_30S = DATA_DIR / 'GenreClassData_30s.txt'


def read_genre_class_data(
    file_path: Path,
):
    if file_path.is_file() is False:
        raise ValueError(f'Given file path is not a file ({file_path})')
    return pandas.read_table(file_path)


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

    audio_data = audio_data[:number_of_segments * samples_per_segment]

    return np.reshape(audio_data, (number_of_segments, samples_per_segment))

