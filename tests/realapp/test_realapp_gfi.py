from pyreal import RealApp


def test_produce_global_feature_importance(regression_no_transforms):
    realApp = RealApp(
        regression_no_transforms["model"],
        regression_no_transforms["x"],
        transformers=regression_no_transforms["transformers"],
    )
    features = ["A", "B", "C"]

    explanation = realApp.produce_global_feature_importance()

    assert list(explanation["Feature Name"]) == features
    assert list(explanation["Importance"]) == [4/3, 0, 0]

    # confirm caching doesn't break
    realApp.produce_global_feature_importance()
