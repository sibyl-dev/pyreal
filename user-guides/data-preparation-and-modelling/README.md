---
description: Introduction to preparing the components of ML decision-making
---

# Data Preparation and Modelling

Pyreal wraps all the components you need for an  ML decision-making use-case in a single RealApp object. This allows you to make model predictions, transform data, get explanations, and otherwise interact with your ML model for decision-making.&#x20;

In this guide, we go through these components and introduce the process of preparing them.&#x20;

{% hint style="info" %}
If you already have a fully set-up ML workflow (including an ML model, data, and possibly data transformers), you can head over to the [Migrating to Pyreal](https://github.com/sibyl-dev/pyreal/blob/dev/tutorials/migrating\_to\_pyreal.ipynb) tutorial to get started.
{% endhint %}

## On this page...

<table data-view="cards"><thead><tr><th></th><th></th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td><h3>Training and Input Data</h3></td><td>Learn to prepare your data as pandas DataFrames</td><td><a href="./#training-and-input-data-1">#training-and-input-data-1</a></td></tr><tr><td><h3>Transformers</h3></td><td>Learn to create Pyreal data transformers for feature engineering </td><td><a href="./#transformers-1">#transformers-1</a></td></tr><tr><td><h3>Modelling</h3></td><td>Learn how to train models to make predictions on your data</td><td><a href="./#modelling-1">#modelling-1</a></td></tr></tbody></table>

## Training and Input Data

Pyreal expects data in the format of [Pandas DataFrames](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). Each row refers to one _data instance_ (a person, place, thing, or entity), and each column refers to a _feature_, or piece of information about that instance. Column headers are the names of feature. Each instance may optionally have an instance ID, which can either be stored as the DataFrame's indices (row IDs) or as a separate column.

For example, a part of your data may look like:

<table><thead><tr><th data-type="number">house_id</th><th data-type="number">size</th><th>location</th><th>garden_size</th></tr></thead><tbody><tr><td>101</td><td>2200</td><td>Pinewood</td><td>100</td></tr><tr><td>102</td><td>1500</td><td>Oceanview</td><td><em>N/A</em></td></tr><tr><td>103</td><td>1800</td><td>Placedale</td><td>120</td></tr></tbody></table>

There are two categories of data relevant to ML decision-making: the training data and the input data.

The training data is used to train the ML model and explainers. The input data is the data that you actively wish to get predictions on and understand better. The main difference between these two types is data is that you usually will have the _ground truth_ values (the "correct" answer for the value your model tries to predict) for your training data but not your input data.

For example, if we are trying to predict house prices, you would have additional information about the price of houses in your training dataset.

<table data-full-width="false"><thead><tr><th data-type="number">house_id</th><th data-type="number">price ($)</th></tr></thead><tbody><tr><td>101</td><td>250000</td></tr><tr><td>102</td><td>220000</td></tr><tr><td>103</td><td>180000</td></tr></tbody></table>

## Transformers

Many ML models either require data to be in a specific format, or preform significantly better when data is a specific format.&#x20;

For example, many models require all data to be numeric, cannot handle missing data, or expect all features to be on similar numeric scales. But this is rarely the case in real-world applications, so we need to perform **feature engineering** using **data transformers**.

In this section, we offer an introduction to a few common transformer types.

{% hint style="info" %}
If this topic is new to you, you may find it helpful to look at more guides like the [sklearn transformer documentation](https://scikit-learn.org/stable/data\_transforms.html).
{% endhint %}

### Types of Transformers

#### Handling categorical features: One-hot encoding

One way to handle categorical features like the Location feature above is with **one-hot encoding**. In this process, we turn a single column into one column per feature value. We set the value-column corresponding to the row's value to True, and all others to False (and then represent these values as True=1, False=0). For example, after one-hot encoding the Location feature, our location features will look like:

<table><thead><tr><th data-type="number">house_id</th><th>location (Pinewood)</th><th>location (Oceanview)</th><th width="232">location (Placedale)</th></tr></thead><tbody><tr><td>101</td><td>1</td><td>0</td><td>0</td></tr><tr><td>102</td><td>0</td><td>1</td><td>0</td></tr><tr><td>103</td><td>0</td><td>0</td><td>1</td></tr></tbody></table>

**With Pyreal** you can one-hot encode data using the **OneHotEncoder** transformer, setting the `columns`  parameter to a list of your categorical columns:

```python
from pyreal.transformers import OneHotEncoder

oh_encoder = OneHotEncoder(columns=["Location"])
```

#### Handling missing data: Imputing

Sometimes, you won't have values for all features for all instances. Maybe you don't know the size of every house's garden, or some houses don't have gardens at all.

In this case, one solution is the **impute** the missing values using the average or most frequent values for each feature. For example, after imputing our sample table, the garden size feature may look like:

<table><thead><tr><th data-type="number">House ID</th><th>Garden size</th></tr></thead><tbody><tr><td>101</td><td>100</td></tr><tr><td>102</td><td><em>110</em></td></tr><tr><td>103</td><td>120</td></tr></tbody></table>

**With Pyreal**, you can impute your data using the **MultiTypeImputer**, which automatically imputes numeric features with the mean (average) value and categorical features with the mode (most frequent) value.

```python
from pyreal.transformers import MultiTypeImputer

imputer = MultiTypeImputer()If you'd like to use alternative imputation 
```

{% hint style="info" %}
If you'd like to use an imputation strategy that is not directly supported in Pyreal (for example, maybe you know that all houses with missing garden sizes don't have gardens, and want to impute their values with 0), you can wrap a transformer option from other libraries such as sklearn in a Pyreal generic Explainer object. For example:

```python
from sklearn.impute import SimpleImputer
from pyreal.transformers import Transformer

imp = Transformer(SimpleImputer(strategy="constant", fill_value=0))
```
{% endhint %}

#### Scaling numeric features: Standardization

Many types of ML models perform best when all features are in similar numeric ranges; otherwise, features will very large values may outweigh features with smaller values, even if they are actually less important. One approach to addressing this is using **standardization**, which scales all feature values to have a mean of 0 and a variance of 1.&#x20;

**With Pyreal**, you can scale with a **StandardScaler:**

```python
from pyreal.transformers import StandardScaler

scaler = StandardScaler()
```

{% hint style="info" %}
Pyreal also supports min-max scaling (scaling between a set minimum and maximum value) and normalizing (setting the l1/l2 norm to 1) using the **MinMaxScaler** and **Normalizer** respectively.&#x20;
{% endhint %}

### Fitting transformers

Most transformers need to be fit to training data, for example to determine categorical feature values or feature ranges/mean values. With Pyreal, you can either fit each transformer manually using the `Transformer.fit(data)` function or fit them all together using the `fit_transformers` function.

{% hint style="warning" %}
The order you fit/run your transformers matters! For example, you will generally want to start by imputing your data, then one-hot encode your categorical features, and then scaling your features (which should all be numeric now).&#x20;

Many transformers will break if they encounter missing values, so you will often need to start by imputing.
{% endhint %}

```python
from pyreal.transformers import (
        fit_transformers, 
        OneHotEncoder, 
        MultiTypeImputer, 
        StandardScaler
)
from pyreal.sample_applications import california_housing 

# Simple sample data (X_train: pandas DataFrame, y_train: pandas Series)
X_train, y_train = california_housing.load_data() 

# Initialize transformers
transformers = [MultiTypeImputer(), 
                OneHotEncoder(columns=["Location"]),
                StandardScaler()]
                
# Fit transformers to training data
fit_transformers(transformers, X_train)
```

{% hint style="info" %}
As with most Pyreal functions, the Transformer.fit(data) (and Transformer.transform(data)) functions take in pandas DataFrames (and in the case of transform, returns a pandas DataFrame).
{% endhint %}

You are now ready to use your transformers with you ML application!

## Modelling

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
