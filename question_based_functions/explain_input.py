"""
Defines top-level explanation methods
"""

model = None  # the model
X = None  # a relevant dataset  *may not be available
X_train = None  # the training dataset  *may not be available

# Why did the model make this prediction?
def explain(x):
    """
    Explains the model's prediction on input datapoint
    :param x: a valid input for model
    :return: an explanation of why model(x) predicts as it does
    """
    return None

# Why didn't the model predict class2?
def explain_counter(x, class2):
    """
    Explains why the model does not predict class2 on x
    :param x: a valid input for model
    :param class2: a class supported by model, class2 != model(x)
    :return: an explanation of why model(x) != class2
    """
    return None

# How sure is the model of this answer?
def uncertainty(x):
    """
    Finds the uncertainty of model(x)
    :param x: a valid input for model
    :return: some uncertainty measure of model(x)
    """
    return None

# How much should I trust this prediction?
def trust_explain(x):
    """
    Provides a analysis of the trustworthiness of the prediction model(x)
    :param x: a valid input for model
    :return: a explanation that addresses the trustworthiness of the prediction model(x)
    """
    return None

# Is this prediction fair?
def fairness_explain(x):
    """
    Provides an analysis of the fairness of the prediction model(x), with regard to
    features protected_features
    :param x: a valid input for model
    :return: an explanation that addresses the trustworthiness of the prediction model(x)
    """
    return None

# How did this prediction get calculated?
def method(x):
    """
    Details the steps taken that led to model(x)
    :param x: a valid input for the model
    :return: a step-by-step explanation of the model's methodology
    """
    return None

# How did you get this value for this feature?
def explain_feature_value(x, feature):
    """
    Returns the method used to get the value of feature
    :param x: a valid input to the model
    :param feature:
    :return: the method used to calculate feature in human readable terms
    """
    return None
