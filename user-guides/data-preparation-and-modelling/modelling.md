# Modelling

One you have your data and transformers ready, you can train an ML model to make predictions.

{% hint style="info" %}
Here we give a quick overview to training ML models with one type of model. If this is a new topic to you, you may find it helpful to look into further guides like the [sklearn user guide](https://scikit-learn.org/stable/user\_guide.html).
{% endhint %}

There are many Python libraries out there for training ML models, such as:

* [sklearn](https://scikit-learn.org/stable/index.html)
* [XGBoost ](https://xgboost.readthedocs.io/en/stable/)
* [Keras](https://keras.io/)
* [Pytorch](https://pytorch.org/)
* [Tensorflow](https://www.tensorflow.org/)

Pyreal can accept any model object with a `.predict()` function that takes in data and returns a prediction.&#x20;

In this guide, we will use XGBoost, an efficient and powerful library that offers classfiers and regressors using the gradient boosting framework. It is an effective choice for many ML use cases.

You will need to install XGBoost, which can be done with pip:

```bash
pip install xgboost
```

### Step 1: Preparing the data

Usually when we train ML models, we split our training data (the data with ground-truth labels) into a _training set_ and _testing set_. The training set is used to fit the ML model, and the testing set is used to validate how well our model performs.&#x20;

We will also need to transform out data to get it ready for the ML model. Typically, we fit our transformers on the training set, and then transform all data using the fit transformers

```python
from sklearn.model_selection import train_test_split
from pyreal.sample_applications import california_housing 
from pyreal.transformers import (
        fit_transformers, 
        run_transformers,
        OneHotEncoder, 
        MultiTypeImputer, 
        StandardScaler
)

# Simple sample data (X_train: pandas DataFrame, y_train: pandas Series)
X, y = california_housing.load_data() 

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Initialize transformers
transformers = [MultiTypeImputer(), 
                OneHotEncoder(columns=["Location"]),
                StandardScaler()]
                
# Fit transformers to training data
fit_transformers(transformers, X_train)

# Transform the data for the model
X_train_model = run_transformers(transformers, X_train)
X_test_model = run_transformers(transformers, X_test)
```

### Step 2: Fitting and evaluating the model

We can now initialize, fit, and evaluate the model's performance on the test data:

```python
from xgboost import XGBRegressor()

# Initialize the model
model = XGBRegressor()

# Fit the model to the data
model.fit(X_train_model, y_train)

# Evaluate the model on the test dataset (outputs R^2 score)
model.score(X_test_model, y_test)
```

The final line above gives you the R^2 score of the model. The closer the score is 1, the better the performance. To improve your performance, you can experiment with the training parameters of the model, or with additional feature engineering options.&#x20;

## Next Steps

You now have all the components needed to start working with an ML application! Continue onto the next page to learn how to set up a Pyreal application and start getting predictions and explanations of your model.
