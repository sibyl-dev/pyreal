from pyreal.explainers import PartialDependence


def test_fit_pdp(all_models):
    for model in all_models:
        simple = PartialDependence(
            model=model["model"], x_train_orig=model["x"], transformers=model["transformers"]
        )
        # Assert no error
        simple.fit()
