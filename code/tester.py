from plotting import *
from data_handling import *
from descriptive_statistics import *

X = read_genre_class_data(GENRE_CLASS_DATA_30S)

vec1 = np.array([8056.984614, 112.347147, 3889.624969])
vec2 = np.array([7633.162177, 89.102909, 3727.466037])
print(np.linalg.norm(vec1 - vec2))

test = X["Track ID"]
print(test)
genres = ["metal", "jazz", "classical", "pop"]
features = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean"]


principal_components_reduction(X, genres)
threed_scatter(X, features=features, genres=genres)

