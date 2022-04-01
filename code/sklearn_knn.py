import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import data_handling as dh


def predict(
    training_data: dh.Dataset,
    test_data: dh.Dataset,
) -> pd.DataFrame:
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X=training_data.x, y=training_data.y)

    return knn_classifier.predict(X=test_data.x)
