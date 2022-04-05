from plotting import *
from data_handling import *
from descriptive_statistics import *
from diy_classifiers import *

X = read_genre_class_data(GENRE_CLASS_DATA_30S)

vec1 = np.array([8056.984614, 112.347147, 3889.624969])
vec2 = np.array([7633.162177, 89.102909, 3727.466037])
print(np.linalg.norm(vec1 - vec2))

train_frame = X[X["Type"] == "Train"]
test_frame = X[X["Type"] == "Test"]

k = 8
genres = ["metal", "jazz", "classical", "pop"]
features = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]

diy_kNN(k, train_frame, test_frame, features, p=4)
#principal_components_reduction(X, genres)
#threed_scatter(X, features=features, genres=genres)

