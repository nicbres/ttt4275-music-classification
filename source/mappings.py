from typing import Union, Iterable

import pandas as pd

GENRES = {
    1: "pop",
    2: "metal",
    3: "disco",
    4: "blues",
    5: "reggae",
    6: "classical",
    7: "rock",
    8: "hiphop",
    9: "country",
    10: "jazz",
}


FEATURES = pd.DataFrame(
    [
        {
            1: "Track ID",
            3: "zero_cross_rate_mean",
            4: "zero_cross_rate_std",
            5: "rmse_mean",
            6: "rmse_var",
            7: "spectral_centroid_mean",
            8: "spectral_centroid_var",
            9: "spectral_bandwidth_mean",
            10: "spectral_bandwidth_var",
            11: "spectral_rolloff_mean",
            12: "spectral_rolloff_var",
            13: "spectral_contrast_mean",
            14: "spectral_contrast_var",
            15: "spectral_flatness_mean",
            16: "spectral_flatness_var",
            17: "chroma_stft_1_mean",
            18: "chroma_stft_2_mean",
            19: "chroma_stft_3_mean",
            20: "chroma_stft_4_mean",
            21: "chroma_stft_5_mean",
            22: "chroma_stft_6_mean",
            23: "chroma_stft_7_mean",
            24: "chroma_stft_8_mean",
            25: "chroma_stft_9_mean",
            26: "chroma_stft_10_mean",
            27: "chroma_stft_11_mean",
            28: "chroma_stft_12_mean",
            29: "chroma_stft_1_std",
            30: "chroma_stft_2_std",
            31: "chroma_stft_3_std",
            32: "chroma_stft_4_std",
            33: "chroma_stft_5_std",
            34: "chroma_stft_6_std",
            35: "chroma_stft_7_std",
            36: "chroma_stft_8_std",
            37: "chroma_stft_9_std",
            38: "chroma_stft_10_std",
            39: "chroma_stft_11_std",
            40: "chroma_stft_12_std",
            41: "tempo",
            42: "mfcc_1_mean",
            43: "mfcc_2_mean",
            44: "mfcc_3_mean",
            45: "mfcc_4_mean",
            46: "mfcc_5_mean",
            47: "mfcc_6_mean",
            48: "mfcc_7_mean",
            49: "mfcc_8_mean",
            50: "mfcc_9_mean",
            51: "mfcc_10_mean",
            52: "mfcc_11_mean",
            53: "mfcc_12_mean",
            54: "mfcc_1_std",
            55: "mfcc_2_std",
            56: "mfcc_3_std",
            57: "mfcc_4_std",
            58: "mfcc_5_std",
            59: "mfcc_6_std",
            60: "mfcc_7_std",
            61: "mfcc_8_std",
            62: "mfcc_9_std",
            63: "mfcc_10_std",
            64: "mfcc_11_std",
            65: "mfcc_12_std",
            66: "GenreID",
            67: "Genre",
            68: "Type",
        }
    ]
)


def get_features_from_indices(
    indices: Union[int, Iterable],
):
    if any([index not in FEATURES.keys() for index in indices]):
        raise ValueError("Indices not available for given features, use {1, 3-68}")
    return list(FEATURES.loc[0, indices])


MUSIC_FEATURES_ALL = get_features_from_indices(range(3, 66))
MUSIC_FEATURES_MEANS = list(
    filter(lambda feature: "mean" in feature, MUSIC_FEATURES_ALL)
)
