from explanation_toolkit.utils import identity_transform


def binary_flip_all(predict, x_orig, features, transform=identity_transform):
    """
    Get the results of flipping each binary variable in an input

    :param predict: the prediction function to use
    :param x_orig: the input to explain, unstandardized
    :param features: the names of all features to swap x_orig
    :param transform: the transformation function to prepare inputs for predict
    :return: a tuple of lists where the first list is the resulting prediction from flipping each binary variable,
             and the second list is the new values of each flip.
    """
    flip_preds = []
    values = []
    for feat in features:
        mod_x = x_orig.copy()
        mod_x[feat] = 1 - x_orig[feat]
        value = 1 - x_orig[feat]
        mod_x = transform(mod_x)
        pred = predict(mod_x.values.reshape(1, -1))
        flip_preds.append(pred[0])
        values.append(value)
    return flip_preds, values


def modify_and_repredict(predict, x_orig, features, new_values, transform=identity_transform):
    """
    Make changes to x and then return the new prediction

    :param predict: the prediction function to use
    :param x_orig: the untransformed input to modify
    :param features: a list of feature names to change
    :param new_values: the new values to give the features
    :param transform: the transformation to use to ready the input for the prediction function
    :return: the new prediction after making the requested changes
    """
    x_new = x_orig.copy()
    x_new[features] = new_values
    x_new = transform(x_new)
    new_pred = predict(x_new.values.reshape(1,-1))
    return new_pred
