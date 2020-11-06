from sklearn.linear_model import LogisticRegression


def logistic_regression(X_, y_):
    model = LogisticRegression().fit(X_, y_)
    return model
