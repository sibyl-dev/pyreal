from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import Normalizer as SklearnNormalizer
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from pyreal.transformers.scale import MinMaxScaler, Normalizer, StandardScaler


def test_fit_minmax(scale_data):
    sk_min_max = SklearnMinMaxScaler()
    sk_min_max.fit(scale_data["ndarray"])

    min_max = MinMaxScaler()
    min_max.fit(scale_data["pandas"])

    # our scaler objects are different types; test the attributes
    assert all(
        [sk_min_max.data_min_[i] == min_max.data_min_[i] for i in range(len(sk_min_max.data_min_))]
    )
    assert all(
        [sk_min_max.data_max_[i] == min_max.data_max_[i] for i in range(len(sk_min_max.data_max_))]
    )
    assert all(
        [
            sk_min_max.data_range_[i] == min_max.data_range_[i]
            for i in range(len(sk_min_max.data_range_))
        ]
    )
    assert all([sk_min_max.min_[i] == min_max.min_[i] for i in range(len(sk_min_max.min_))])


def test_inverse_transform_minmax(scale_data):
    sklearn_scaler = SklearnMinMaxScaler()
    sklearn_scaler.fit(scale_data["ndarray"])
    sk = sklearn_scaler.inverse_transform(scale_data["ndarray"])

    min_max = MinMaxScaler()
    min_max.fit(scale_data["pandas"])
    mm = min_max.inverse_transform(scale_data["pandas"])

    for i in range(len(sk)):
        assert all([sk[i][j] == mm.values[i][j] for j in range(len(sk[i]))])


def test_transform_minmax(scale_data):
    sk_min_max = SklearnMinMaxScaler()
    sk_min_max.fit(scale_data["ndarray"])
    sk = sk_min_max.transform(scale_data["ndarray"])

    min_max = MinMaxScaler()
    min_max.fit(scale_data["pandas"])
    mm = min_max.data_transform(scale_data["pandas"])
    for j in range(len(sk)):
        assert all([sk[j][i] == mm.values[j][i] for i in range(len(sk[0]))])


# NORMALIZER TESTS


def test_fit_normalizer(scale_data):
    norm = Normalizer()
    norm.fit(scale_data["pandas"])
    # if nothing happens, it passes


def test_transform_normalizer(scale_data):
    sk_normal = SklearnNormalizer()
    sk_normal.fit(scale_data["ndarray"])
    sk = sk_normal.transform(scale_data["ndarray"])

    normal = Normalizer()
    normal.fit(scale_data["pandas"])
    norm = normal.data_transform(scale_data["pandas"])

    for i in range(len(sk)):
        assert (sk[i][j] == norm.values[i][j] for j in range(len(sk[i])))


# STANDARD SCALER


def test_fit_standardscale(scale_data):
    sk_standard = SklearnStandardScaler()
    sk_standard.fit(scale_data["ndarray"])

    standard = StandardScaler()
    standard.fit(scale_data["pandas"])

    assert (sk_standard.mean_[i] == standard.mean_[i] for i in range(len(sk_standard.mean_)))
    assert (sk_standard.var_[i] == standard.var_[i] for i in range(len(sk_standard.var_)))


def test_transform_standardscale(scale_data):
    sk_standard = SklearnStandardScaler()
    sk_standard.fit(scale_data["ndarray"])
    sk = sk_standard.transform(scale_data["ndarray"])

    standard = StandardScaler()
    standard.fit(scale_data["pandas"])
    std = standard.data_transform(scale_data["pandas"])

    for i in range(len(sk)):
        assert (sk[i][j] == std.values[i][j] for j in range(len(sk[i])))


def test_inverse_transform_standardscale(scale_data):
    sklearn_scaler = SklearnStandardScaler()
    sklearn_scaler.fit(scale_data["ndarray"])
    sk = sklearn_scaler.inverse_transform(scale_data["ndarray"])

    standard = StandardScaler()
    standard.fit(scale_data["pandas"])
    std = standard.inverse_transform(scale_data["pandas"])

    for i in range(len(sk)):
        assert all([sk[i][j] == std.values[i][j] for j in range(len(sk[i]))])


if __name__ == "__main__":
    test_fit_standardscale()
