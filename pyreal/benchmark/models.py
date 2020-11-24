from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense


def logistic_regression(X, y):
    model = LogisticRegression().fit(X, y)
    return model


def small_neural_network(X, y):
    model = Sequential()
    input_dim = X.shape[1]
    if y.ndim == 1:
        output_dim = 1
        loss = "mean_squared_error"
    else:
        output_dim = y.shape[1]
        loss = "binary_crossentropy"
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dense(output_dim), activation='sigmoid')
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    model.fix(X, y, epochs=10)
