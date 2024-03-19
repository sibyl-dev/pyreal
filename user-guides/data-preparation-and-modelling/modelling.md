# Modeling

{% hint style="info" %}
The full code for this and all other user guides can be found in our [user guide tutorial](https://github.com/sibyl-dev/pyreal/blob/dev/tutorials/user\_guide.ipynb).
{% endhint %}

Once you have your data and transformers ready, you can train an ML model to make predictions.

{% hint style="info" %}
In this guide we give a quick overview to training ML models with one type of model. If this is a new topic to you, you may find it helpful to look into further guides like the [sklearn user guide](https://scikit-learn.org/stable/user\_guide.html).
{% endhint %}

There are many Python libraries out there for training ML models, such as:

* [sklearn](https://scikit-learn.org/stable/index.html)
* [XGBoost ](https://xgboost.readthedocs.io/en/stable/)
* [LightGBM](https://lightgbm.readthedocs.io/en/stable/)
* [Keras](https://keras.io/)
* [Pytorch](https://pytorch.org/)
* [Tensorflow](https://www.tensorflow.org/)

Pyreal can accept any model object with a `.predict()` function that takes in data and returns a prediction.&#x20;

In this guide, we will use LightGBM, a powerful and lightweight library that offers classfiers and regressors using the gradient boosting framework. It is an effective choice for many ML use cases.

You may need to install LightGBM, which can be done with pip:

```bash
pip install lightgbm
```

### Step 1: Preparing the data

As described in the data guide, usually when we train ML models we split our training data (the data with ground-truth labels) into a _training set_ and _testing set_. The training set is used to fit the ML model, and the testing set is used to validate how well our model performs.&#x20;

We will also need to transform out data to get it ready for the ML model. We can do this using our transformers we fit in the Transformers guide.

```python
from pyreal.transformers import run_transformers

# Transform the data for the model
x_train_model = run_transformers(transformers, x_train)
x_test_model = run_transformers(transformers, x_test)
```

### Step 2: Fitting and evaluating the model

We can now initialize, fit, and evaluate the model's performance on the test data:

```python
from lightgbm import LGBMRegressor

# Initialize the model
model = LGBMRegressor()

# Fit the model to the data
model.fit(x_train_model, y_train)

# Evaluate the model on the test dataset (outputs R^2 score)
model.score(x_test_model, y_test)
```

The final line above gives you the R^2 score of the model. The closer the score is 1, the better the performance. To improve your performance, you can experiment with the training parameters of the model, or with additional feature engineering options.&#x20;

## Next Steps

You now have all the components needed to start working with an ML application! Continue onto the next page to learn how to set up a Pyreal application and start getting predictions and explanations of your model.
