from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import Normalizer as SklearnNormalizer
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from pyreal.transformers.scale import MinMaxScaler, Normalizer, StandardScaler

# MinMaxScaler tests
data = [[2, 1, 3, 10], [4, 3, 4, 0], [6, 7, 2, 2]]
columns = ["A", "B", "C", "D"]

# test whether pyreal and sklearn scalers behave the same


def test_fit_minmax():
    skMinMax = SklearnMinMaxScaler()
    skMinMax.fit(data)

    pdData = DataFrame(data)
    minmax = MinMaxScaler()
    minmax.fit(pdData)

    # our scaler objects are different types; test the attributes
    assert all(
        [skMinMax.data_min_[i] == minmax.data_min_[i] for i in range(len(skMinMax.data_min_))]
    )
    assert all(
        [skMinMax.data_max_[i] == minmax.data_max_[i] for i in range(len(skMinMax.data_max_))]
    )
    assert all(
        [
            skMinMax.data_range_[i] == minmax.data_range_[i]
            for i in range(len(skMinMax.data_range_))
        ]
    )
    assert all([skMinMax.min_[i] == minmax.min_[i] for i in range(len(skMinMax.min_))])


def test_inverse_transform_minmax():
    sklearnScaler = SklearnMinMaxScaler()
    sklearnScaler.fit(data)
    sk = sklearnScaler.inverse_transform(data)

    pdData = DataFrame(data)
    minmax = MinMaxScaler()
    minmax.fit(pdData)
    mm = minmax.inverse_transform(pdData)

    for i in range(len(sk)):
        assert all([sk[i][j] == mm.values[i][j] for j in range(len(sk[i]))])


def test_transform_minmax():
    skMinMax = SklearnMinMaxScaler()
    skMinMax.fit(data)
    sk = skMinMax.transform(data)

    pdData = DataFrame(data)
    minmax = MinMaxScaler()
    minmax.fit(pdData)
    mm = minmax.data_transform(pdData)
    for j in range(len(sk)):
        assert all([sk[j][i] == mm.values[j][i] for i in range(len(sk[0]))])


# NORMALIZER TESTS


def test_fit_normalizer():
    norm = Normalizer()
    pdData = DataFrame(data)
    norm.fit(pdData)
    # if nothing happens, it passes


def test_transform_normalizer():
    skNormal = SklearnNormalizer()
    skNormal.fit(data)
    sk = skNormal.transform(data)

    pdData = DataFrame(data)
    normal = Normalizer()
    normal.fit(pdData)
    norm = normal.data_transform(pdData)

    for i in range(len(data)):
        assert (sk[i][j] == norm.values[i][j] for j in range(len(data[i])))


# STANDARD SCALER


def test_fit_standardscale():
    skStandard = SklearnStandardScaler()
    skStandard.fit(data)

    pdData = DataFrame(data)
    standard = StandardScaler()
    standard.fit(pdData)

    assert (skStandard.mean_[i] == standard.mean_[i] for i in range(len(skStandard.mean_)))
    assert (skStandard.var_[i] == standard.var_[i] for i in range(len(skStandard.var_)))


def test_transform_standardscale():
    skStandard = SklearnStandardScaler()
    skStandard.fit(data)
    sk = skStandard.transform(data)

    pdData = DataFrame(data)
    standard = StandardScaler()
    standard.fit(pdData)
    std = standard.data_transform(pdData)

    for i in range(len(data)):
        assert (sk[i][j] == std.values[i][j] for j in range(len(data[i])))


def test_inverse_transform_standardscale():
    sklearnScaler = SklearnStandardScaler()
    sklearnScaler.fit(data)
    sk = sklearnScaler.inverse_transform(data)

    pdData = DataFrame(data)
    minmax = StandardScaler()
    minmax.fit(pdData)
    mm = minmax.inverse_transform(pdData)

    for i in range(len(sk)):
        assert all([sk[i][j] == mm.values[i][j] for j in range(len(sk[i]))])


if __name__ == "__main__":
    test_fit_standardscale()
