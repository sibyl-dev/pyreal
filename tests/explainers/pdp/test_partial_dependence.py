import pandas as pd

from pyreal.explainers import PartialDependence, PartialDependenceExplainer


def test_produce_partial_dependence_regression_no_transforms(regression_no_transforms):
    model = regression_no_transforms
    features = model["x"].columns[:-1]
    pdpe = PartialDependenceExplainer(
        model=model["model"],
        x_train_orig=model["x"],
        features=features,
        transformers=model["transformers"],
        fit_on_init=True,
    )
    pdp = PartialDependence(
        model=model["model"],
        x_train_orig=model["x"],
        features=features,
        transformers=model["transformers"],
        fit_on_init=True,
    )

    helper_partial_dependence_regression_no_transforms(pdpe, model)
    helper_partial_dependence_regression_no_transforms(pdp, model)


def helper_partial_dependence_regression_no_transforms(explainer, model):
    pdp_object = explainer.produce().get()
    if isinstance(pdp_object.feature_names, pd.Index):
        assert pdp_object.feature_names.tolist() == model["x"].columns[:-1].tolist()
    else:
        assert pdp_object.feature_names == model["x"].columns[:-1]

    for i, dim in enumerate(pdp_object.predictions[0].shape):
        assert dim == len(pdp_object.grid[i])
