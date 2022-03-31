import logging

import librosa
import numpy as np
import pandas


_GENRE_LOOKUP = {
# forward lookup
    'pop': 1,
    'metal': 2,
    'disco': 3,
    'blues': 4,
    'reggae': 5,
    'classical': 6,
    'rock': 7,
    'hiphop': 8,
    'country': 9,
    'jazz': 10,
# reverse lookup
    1: 'pop',
    2: 'metal',
    3: 'disco',
    4: 'blues',
    5: 'reggae',
    6: 'classical',
    7: 'rock',
    8: 'hiphop',
    9: 'country',
    10: 'jazz',
}


_DATA_FRAME_KEYS = [
    'Track ID', 'zero_cross_rate_mean', 'zero_cross_rate_std', 'rmse_mean',
    'rmse_var', 'spectral_centroid_mean', 'spectral_centroid_var',
    'spectral_bandwidth_mean', 'spectral_bandwidth_var',
    'spectral_rolloff_mean', 'spectral_rolloff_var',
    'spectral_contrast_mean', 'spectral_contrast_var',
    'spectral_flatness_mean', 'spectral_flatness_var', 'chroma_stft_1_mean',
    'chroma_stft_2_mean', 'chroma_stft_3_mean', 'chroma_stft_4_mean',
    'chroma_stft_5_mean', 'chroma_stft_6_mean', 'chroma_stft_7_mean',
    'chroma_stft_8_mean', 'chroma_stft_9_mean', 'chroma_stft_10_mean',
    'chroma_stft_11_mean', 'chroma_stft_12_mean', 'chroma_stft_1_std',
    'chroma_stft_2_std', 'chroma_stft_3_std', 'chroma_stft_4_std',
    'chroma_stft_5_std', 'chroma_stft_6_std', 'chroma_stft_7_std',
    'chroma_stft_8_std', 'chroma_stft_9_std', 'chroma_stft_10_std',
    'chroma_stft_11_std', 'chroma_stft_12_std', 'tempo', 'mfcc_1_mean',
    'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean',
    'mfcc_6_mean', 'mfcc_7_mean', 'mfcc_8_mean', 'mfcc_9_mean',
    'mfcc_10_mean', 'mfcc_11_mean', 'mfcc_12_mean', 'mfcc_1_std',
    'mfcc_2_std', 'mfcc_3_std', 'mfcc_4_std', 'mfcc_5_std', 'mfcc_6_std',
    'mfcc_7_std', 'mfcc_8_std', 'mfcc_9_std', 'mfcc_10_std', 'mfcc_11_std',
    'mfcc_12_std', 'GenreID', 'Genre', 'Type',
]

_FUNCTION_OUTPUT_SHAPE = {
    librosa.feature.zero_crossing_rate: (1,),
    librosa.feature.rms: (1,),
    librosa.feature.spectral_centroid: (1,),
    librosa.feature.spectral_bandwidth: (1,),
    librosa.feature.spectral_rolloff: (1,),
    librosa.feature.spectral_contrast: (7,),
    librosa.feature.spectral_flatness: (1,),
    librosa.feature.chroma_stft: (12,),
    librosa.beat.tempo: (1,),
    librosa.feature.mfcc: (20,),
}


def extract_audio_features(
    track_id: int,
    genre: str,
    audio_data: np.ndarray,
    sample_rate: float,
    set_type: str = "train",
):
    audio_data = np.atleast_2d(audio_data)

    genre_id = _GENRE_LOOKUP[genre]

    zcr_mean, zcr_std = get_zero_crossing_rate(
        audio_data=audio_data,
    )
    logging.debug(f"Zero Crossing Rate Mean:\t{np.shape(zcr_mean)}")
    logging.debug(f"Zero Crossing Rate Std:\t{np.shape(zcr_std)}")

    rmse_mean, rmse_var = get_rmse(
        audio_data=audio_data,
    )
    logging.debug(f"RMSE Mean:\t{np.shape(rmse_mean)}")
    logging.debug(f"RMSE Std:\t{np.shape(rmse_var)}")

    spectral_centroid_mean, spectral_centroid_var = get_spectral_centroid(
        audio_data=audio_data,
        sample_rate=sample_rate,
    )
    logging.debug(f"Spectral Centroid Mean:\t{np.shape(spectral_centroid_mean)}")
    logging.debug(f"Spectral Centroid Std:\t{np.shape(spectral_centroid_var)}")

    spectral_bandwidth_mean, spectral_bandwidth_var = get_spectral_bandwidth(
        audio_data=audio_data,
        sample_rate=sample_rate,
    )
    logging.debug(f"Spectral Bandwidth Mean:\t{np.shape(spectral_bandwidth_mean)}")
    logging.debug(f"Spectral Bandwidth Std:\t{np.shape(spectral_bandwidth_var)}")

    spectral_rolloff_mean, spectral_rolloff_var = get_spectral_rolloff(
        audio_data=audio_data,
        sample_rate=sample_rate,
    )
    logging.debug(f"Spectral Rolloff Mean:\t{np.shape(spectral_rolloff_mean)}")
    logging.debug(f"Spectral Rolloff Std:\t{np.shape(spectral_rolloff_var)}")

    spectral_contrast_mean, spectral_contrast_var = get_spectral_contrast(
        audio_data=audio_data,
        sample_rate=sample_rate,
    )
    logging.debug(f"Spectral Contrast Mean:\t{np.shape(spectral_contrast_mean)}")
    logging.debug(f"Spectral Contrast Std:\t{np.shape(spectral_contrast_var)}")

    spectral_flatness_mean, spectral_flatness_var = get_spectral_flatness(
        audio_data=audio_data,
    )
    logging.debug(f"Spectral Flatness Mean:\t{np.shape(spectral_flatness_mean)}")
    logging.debug(f"Spectral Flatness Std:\t{np.shape(spectral_flatness_var)}")

    chroma_stft_mean, chroma_stft_std = get_chroma_stft(
        audio_data=audio_data,
        sample_rate=sample_rate,
    )
    logging.debug(f"Chroma STFT Mean:\t{np.shape(chroma_stft_mean)}")
    logging.debug(f"Chroma STFT Std:\t{np.shape(chroma_stft_std)}")

    tempo = get_tempo(
        audio_data=audio_data,
        sample_rate=sample_rate,
    )
    logging.debug(f"Tempo:\t{np.shape(tempo)}")

    mfcc_mean, mfcc_std = get_mfcc(
        audio_data=audio_data,
        sample_rate=sample_rate,
    )
    logging.debug(f"MFCC Mean:\t{np.shape(mfcc_mean)}")
    logging.debug(f"MFCC Std:\t{np.shape(mfcc_std)}")

    entry = np.concatenate(
        (
            np.atleast_2d(np.full(np.shape(audio_data)[0], track_id)),
            zcr_mean,
            zcr_std,
            rmse_mean,
            rmse_var,
            spectral_centroid_mean,
            spectral_centroid_var,
            spectral_bandwidth_mean,
            spectral_bandwidth_var,
            spectral_rolloff_mean,
            spectral_rolloff_var,
            spectral_contrast_mean,
            spectral_contrast_var,
            spectral_flatness_mean,
            spectral_flatness_var,
            chroma_stft_mean,
            chroma_stft_std,
            tempo,
            mfcc_mean,
            mfcc_std,
            np.atleast_2d(np.full(np.shape(audio_data)[0], genre_id)),
            np.atleast_2d(np.full(np.shape(audio_data)[0], genre)),
            np.atleast_2d(np.full(np.shape(audio_data)[0], set_type)),
        ),
        axis=1,
    )

    data_frame = pandas.DataFrame(
        entry,
        columns=_DATA_FRAME_KEYS,
    )

    return data_frame


def get_mean_and_moment(
    function,
    audio_data: np.ndarray,
    sample_rate: float = None,
    moment_function = np.std,
):
    audio_data = np.atleast_2d(audio_data)
    function_output_shape = (np.shape(audio_data)[0], _FUNCTION_OUTPUT_SHAPE[function][0])

    mean = np.zeros(function_output_shape)
    moment = np.zeros(function_output_shape)

    for index, row in enumerate(audio_data):
        if sample_rate:
            value = function(
                y=row,
                sr=sample_rate,
            )
        else:
            value = function(
                y=row,
            )

        mean[index] = np.mean(value, axis=1)
        moment[index] = moment_function(value, axis=1)

    return mean, moment


def get_zero_crossing_rate(
    audio_data: np.ndarray,
):
    zero_crossing_mean, zero_crossing_std = get_mean_and_moment(
        function=librosa.feature.zero_crossing_rate,
        audio_data=audio_data,
    )

    return zero_crossing_mean, zero_crossing_std


def get_rmse(
    audio_data: np.ndarray,
):
    rmse_mean, rmse_var = get_mean_and_moment(
        function=librosa.feature.rms,
        moment_function=np.var,
        audio_data=audio_data,
    )

    return rmse_mean, rmse_var


def get_spectral_centroid(
    audio_data: np.ndarray,
    sample_rate: float,
):
    spectral_centroid_mean, spectral_centroid_var = get_mean_and_moment(
        function=librosa.feature.spectral_centroid,
        moment_function=np.var,
        audio_data=audio_data,
        sample_rate=sample_rate,
    )

    return spectral_centroid_mean, spectral_centroid_var


def get_spectral_bandwidth(
    audio_data: np.ndarray,
    sample_rate: float,
):
    spectral_bandwidth_mean, spectral_bandwidth_var = get_mean_and_moment(
        function=librosa.feature.spectral_bandwidth,
        moment_function=np.var,
        audio_data=audio_data,
        sample_rate=sample_rate,
    )

    return spectral_bandwidth_mean, spectral_bandwidth_var


def get_spectral_rolloff(
    audio_data: np.ndarray,
    sample_rate: float,
):
    spectral_rolloff_mean, spectral_rolloff_var = get_mean_and_moment(
        function=librosa.feature.spectral_rolloff,
        moment_function=np.var,
        audio_data=audio_data,
        sample_rate=sample_rate,
    )

    return spectral_rolloff_mean, spectral_rolloff_var


def get_spectral_contrast(
    audio_data: np.ndarray,
    sample_rate: float,
):
    spectral_contrast_mean, spectral_contrast_var = get_mean_and_moment(
        function=librosa.feature.spectral_contrast,
        moment_function=np.var,
        audio_data=audio_data,
        sample_rate=sample_rate,
    )
    
    spectral_contrast_mean = np.atleast_2d(spectral_contrast_mean[:,0])
    spectral_contrast_var = np.atleast_2d(spectral_contrast_var[:,0])

    return spectral_contrast_mean, spectral_contrast_var


def get_spectral_flatness(
    audio_data: np.ndarray,
):
    spectral_flatness_mean, spectral_flatness_var = get_mean_and_moment(
        function=librosa.feature.spectral_flatness,
        moment_function=np.var,
        audio_data=audio_data,
    )

    return spectral_flatness_mean, spectral_flatness_var


def get_chroma_stft(
    audio_data: np.ndarray,
    sample_rate: float,
):
    chroma_stft_mean, chroma_stft_std = get_mean_and_moment(
        function=librosa.feature.chroma_stft,
        audio_data=audio_data,
        sample_rate=sample_rate,
    )

    return chroma_stft_mean, chroma_stft_std


def get_tempo(
    audio_data: np.ndarray,
    sample_rate: float,
):
    audio_data = np.atleast_2d(audio_data)
    tempo = np.zeros(np.shape(audio_data)[0])

    for index, row in enumerate(audio_data):
        oenv = librosa.onset.onset_strength(
            y=row,
            sr=sample_rate,
        )

        tempo[index] = librosa.beat.tempo(
            onset_envelope=oenv,
            sr=sample_rate,
        )[0]

    return np.atleast_2d(tempo)


def get_mfcc(
    audio_data: np.ndarray,
    sample_rate: float,
):
    mfcc_mean, mfcc_std = get_mean_and_moment(
        function=librosa.feature.mfcc,
        audio_data=audio_data,
        sample_rate=sample_rate,
    )

    return mfcc_mean[:,:12], mfcc_std[:,:12]

