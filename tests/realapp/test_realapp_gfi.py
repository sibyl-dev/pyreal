from pyreal import RealApp


def test_produce_global_feature_importance(regression_no_transforms):
    realApp = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        y_train=regression_no_transforms["y"],
        transformers=regression_no_transforms["transformers"],
    )
    features = ["A", "B", "C"]

    explanation = realApp.produce_feature_importance()

    assert list(explanation["Feature Name"]) == features
    assert list(explanation["Importance"]) == [4 / 3, 0, 0]

    explanation = realApp.produce_feature_importance(algorithm="permutation")
    assert list(explanation["Feature Name"]) == features
    assert abs(list(explanation["Importance"])[0]) > 0.1
    assert list(explanation["Importance"])[1:] == [0, 0]

    # confirm no bug in explainer caching
    realApp.produce_feature_importance(algorithm="permutation")


def test_produce_global_feature_importance_no_data_on_init(regression_no_transforms):
    realApp = RealApp(
        regression_no_transforms["model"],
        transformers=regression_no_transforms["transformers"],
    )
    features = ["A", "B", "C"]

    explanation = realApp.produce_feature_importance(
        x_train_orig=regression_no_transforms["x"], y_train=regression_no_transforms["y"]
    )

    assert list(explanation["Feature Name"]) == features
    assert list(explanation["Importance"]) == [4 / 3, 0, 0]

    explanation = realApp.produce_feature_importance(
        algorithm="permutation",
        x_train_orig=regression_no_transforms["x"],
        y_train=regression_no_transforms["y"],
    )
    assert list(explanation["Feature Name"]) == features
    assert abs(list(explanation["Importance"])[0]) > 0.1
    assert list(explanation["Importance"])[1:] == [0, 0]

    # confirm no bug in explainer caching
    realApp.produce_feature_importance(algorithm="permutation")
