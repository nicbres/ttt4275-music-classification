import source.data_handling
import source.descriptive_statistics


def test_given_all_features_and_genres_then_pcr_plot():
    data = source.data_handling.read_genre_class_data(
        file_path=source.data_handling.GENRE_CLASS_DATA_30S,
    )

    source.descriptive_statistics.principal_components_reduction_plot(
        X=data,
    )


def test_given_blues_and_country_all_features_then_pcr_plot():
    genres=["blues", "country"]

    data = source.data_handling.read_genre_class_data(
        file_path=source.data_handling.GENRE_CLASS_DATA_30S,
    )

    data = source.data_handling.reduce_genres(
        data_frame=data,
        genres=genres,
    )

    source.descriptive_statistics.principal_components_reduction_plot(
        X=data,
        genres=genres,
    )
