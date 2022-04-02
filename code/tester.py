from plotting import *
from data_handling import *
from descriptive_statistics import *

X = read_genre_class_data(GENRE_CLASS_DATA_5S)

genres = ["metal", "jazz", "classical", "pop"]
features = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean"]


principal_components_reduction(X, genres)
threed_scatter(X, features=features, genres=genres)

