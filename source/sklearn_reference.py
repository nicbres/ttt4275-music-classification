import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import source.data_handling as dh


class KNN:
    @staticmethod
    def predict(
        training_data: dh.Dataset,
        test_data: dh.Dataset,
        k=5,
        p=2
    ) -> pd.DataFrame:
        knn_classifier = KNeighborsClassifier(n_neighbors=k, p=p)
        knn_classifier.fit(X=training_data.x, y=training_data.y)

        return knn_classifier.predict(X=test_data.x)

