from pyreal.explainers import PartialDependenceExplainer


def test_produce_with_renames(regression_no_transforms):
    model = regression_no_transforms
    transforms = model["transformers"]
    feature_descriptions = {"A": "Feature A", "B": "Feature B"}
    pdp = PartialDependenceExplainer(
        model=model["model"],
        x_train_orig=model["x"],
        fit_on_init=True,
        transformers=transforms,
        features=["A", "B"],
        interpretable_features=True,
        feature_descriptions=feature_descriptions,
    )

    pdp_explanation = pdp.produce().get()
    assert pdp_explanation.feature_names == ["Feature A", "Feature B"]


def test_produce_with_renames_with_size(regression_no_transforms_big):
    model = regression_no_transforms_big
    transforms = model["transformers"]
    feature_descriptions = {"B": "Feature B", "C": "Feature C"}
    pdp = PartialDependenceExplainer(
        model=model["model"],
        x_train_orig=model["x"],
        fit_on_init=True,
        transformers=transforms,
        features=["B", "C"],
        interpretable_features=True,
        feature_descriptions=feature_descriptions,
        training_size=5,
    )

    pdp_explanation = pdp.produce().get()
    assert pdp_explanation.feature_names == ["Feature B", "Feature C"]
    assert pdp_explanation.predictions[0].shape == (100, 100)
