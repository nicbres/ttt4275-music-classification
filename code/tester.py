from plotting import *
from data_handling import *

X = read_genre_class_data(GENRE_CLASS_DATA_30S)

genres = ["blues", "metal", "jazz", "classical", "rock"]
features = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean"]

threed_scatter(X, features=features, genres=genres)