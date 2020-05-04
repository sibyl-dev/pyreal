"""
Methods that allow for detailed insights into the model.
"""


# Can you show me examples where the model was incorrect/correct?
def model_prediction_examples(showCorrect, N, class1 = None, class2 = None):
    """
    Returns examples where the model was correct or incorrect.
    :param showCorrect: True if the user would like examples where the model was correct, False otherwise
    :param N: the number of examples to show
    :param class1: The true class of examples
    :param class2: The predicted class of examples (should be None if showCorrect is True)
    :return: a set of examples that match the criteria
    """
    return None

# Can you tell me about the interactions between these features?
def feature_interactions(features):
    """
    Describes the interactions between some set of features
    :param features: a set of features
    :return: a description of feature interactions
    """
    return None

# Can you should me some examples of what the model would think of as typical examples of a class?
def prototype(class1, N):
    """
    Generates and returns some prototypical examples of class1. The examples should be diverse enough to be informative
    :param class1: the class to show
    :param N: the number of prototypes to show
    :return: the prototypes
    """
    return None

# How much do the features to this model generally affect the output?
def feature_contributions():
    """
    Calculates and returns the overall contributions of all features.
    :return: the contributions of features
    """
    return None


