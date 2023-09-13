import numpy as np


def get_top_contributors(explanation, num_features=5, select_by="absolute"):
    """
    Extracts the top `num_features` most important or contributing features from a feature-based
    explanation.

    Args:
        explanation (DataFrame with an Importance or Contribution column):
            The explanation to extract from
        num_features (int, optional):
            Number of features to extract. Defaults to 5.
        select_by (one of "absolute", "max", "min", optional):
            If absolute, extract the highest importance/contribution by absolute value.
            In max/min, extract the highest/lowest features. Defaults to "absolute".
    """
    if "Contribution" in explanation:
        contributions = explanation["Contribution"]
    elif "Importance" in explanation:
        contributions = explanation["Importance"]
    else:
        raise ValueError("Provided DataFrame has neither Contribution nor Importance column")

    contributions = contributions.to_numpy()
    order = None
    if select_by == "min":
        order = np.argsort(contributions)
    if select_by == "max":
        order = np.argsort(contributions)[::-1]
    if select_by == "absolute":
        order = np.argsort(abs(contributions))[::-1]

    if order is None:
        raise ValueError(
            "Invalid select_by option %s, should be one of 'min', 'max', 'absolute'" % select_by
        )

    return explanation.iloc[order[0:num_features]]
