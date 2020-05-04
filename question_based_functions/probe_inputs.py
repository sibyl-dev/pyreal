"""
Methods that allow for detailed insights into model predictions.
"""

# How would the prediction change under these conditions?
# Can you show me an adversarial attack for this input?
def counterfactual(x, new_class=None, features=None, feature_values=None):
    """
    Returns the smallest possible change, within specifications, to make the desired change.
    :param x: a valid input to the model
    :param new_class:
    :param features:
    :param feature_values:
    :return: a dictionary of {feature_name:new_value} that includes all changed features.
    """
    return None

# Which features were most important in making this decision?
# How much did this feature matter in this decision?
def feature_contribution(x, feature=None):
    """
    Returns the relative contributions of all features, in sorted order.
    :param x: a valid input to the model
    :return: a dictionary of {feature_name:contribution} in descending sorted order by contribution
    """
    return None

# Can you show me examples from the training set that contributed to this prediction?
def influence_examples(x, N):
    """
    Returns N influence examples from the training set for the input x
    :param x: a valid input to the model
    :param N: the number of examples to return
    :return: a set of training examples, including their true values, that are influence examples
    """
    return None

# Can you show me some examples are similar to this input, and how the model responded?
def nearest_neighbors(x, N):
    """
    Finds the N nearest neighbors to the input x and the model's response.
    :param x: a valid input to the model
    :param N: the number of examples to return
    :return: a set of training examples that are most similar to x, and the model's response.
    """
    return None

