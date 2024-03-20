---
description: Getting set up with Pyreal using your own data and ML model.
---

# Your own application

{% hint style="info" %}
The full code for this guide can be found in our [user guide tutorial](https://github.com/sibyl-dev/pyreal/blob/dev/tutorials/user\_guide.ipynb).
{% endhint %}

In the previous Quickstart guide, you learned how to use RealApps to use and understand your ML models. In this guide, we will cover the basics of creating a RealApp for your own application.

Everything covered in this quickstart is discussed in more detail in our user guides.

## Problem setup

In this guide, we will be using data about houses, and looking at an ML model that predicts the price of houses.

## Data Preparation and Modeling

### Data

Pyreal expects data in the format of Pandas DataFrames. Each row refers to one data instance (a person, place, thing, or entity), and each column refers to a feature, or piece of information about that instance. Column headers are the names of feature. Each instance may optionally have an instance ID, which can either be stored as the DataFrame's indices (row IDs) or as a separate column.

In the code below, we load in some data and split it into training and test sets

```python
from pyreal.sample_applications import ames_housing_small
from sklearn.model_selection import train_test_split

x, y = ames_housing_small.load_data(include_targets=True)
x_train, x_test, y_train, y_test = train_test_split(x, y)

x_train.head()
```

Our sample dataset looks like:

| LotArea | Neighborhood | OverallQuality | YearBuilt | Material     | BasementSize | CentralAir |
| ------- | ------------ | -------------- | --------- | ------------ | ------------ | ---------- |
| 12589.0 | Gilbert      | 6              | 2005      | Vinyl Siding | 728.0        | True       |
| 9100.0  | Brookside    | 5              | _missing_ | Vinyl Siding | 944.0        | True       |
| 10125.0 | Mitchell     | _missing_      | 1977      | Plywood      | 483.0        | False      |

### Transformers

RealApps expect a list of Pyreal transformers, which they use to prepare the data for making model predictions. By passing in original data and these transformers separately, you can get explanations of the data presented in the original, understandable and non-transformed format.

To prepare the data for the model, we need to _one-hot encode_ categorical features (or, replace features that take a set number of string category values with a series of Boolean features, one per category), impute features to fill in the missing values, and scale features so they are all on the same numeric scale.

```python
from pyreal.transformers import (OneHotEncoder, 
                                 MultiTypeImputer, 
                                 StandardScaler, 
                                 fit_transformers)

oh_encoder = OneHotEncoder(columns=["Neighborhood", "Material"], 
                           handle_unknown="ignore")
imputer = MultiTypeImputer()
scaler = StandardScaler()

transformers = [oh_encoder, imputer, scaler]
fit_transformers(transformers, x_train).head()
```

### Modeling

We can now transform our training and testing data, and initialize, train, and evaluate our ML model.

In this guide, we will use LightGBM, a powerful and lightweight library that offers classfiers and regressors using the gradient boosting framework. It is an effective choice for many ML use cases.

```python
from pyreal.transformers import run_transformers
from lightgbm import LGBMRegressor

x_train_model = run_transformers(transformers, x_train)
x_test_model = run_transformers(transformers, x_test)

model = LGBMRegressor().fit(x_train_model, y_train)

model.score(x_test_model, y_test)
```

## Creating and Using your RealApp

Creating a RealApp is easy once you have the required components. We will add two additional inputs to make our outputs easier to read: a dictionary of feature names (our data column names) to readable descriptions, and a format function that converts floats to formatted dollar amounts.

```python
from pyreal import RealApp

feature_descriptions = {'LotArea': 'Lot size in square feet', 
                        'Neighborhood': 'Neighborhood', 
                        'OverallQuality': 'Overall quality of the house finishing and materials (1-10)', 
                        'YearBuilt': 'Original construction date', 
                        'Material': 'Exterior material of house', 
                        'BasementSize': 'Total basement area in square feet', 
                        'CentralAir': 'Central air conditioning', 
                        'HouseSize': 'Total above ground living area in square feet'}

realapp = RealApp(model, 
                  X_train_orig=x_train, 
                  y_train=y_train, 
                  transformers=transformers,
                  id_column="House ID",
                  feature_descriptions=feature_descriptions,
                  pred_format_func=lambda x: f"${x:,.2f}")
```

You can now use the `.predict` and `.produce` functions to use and understand your ML model

```python
predictions = realapp.predict(x_input)

print(f"Predicted price for House 101: {predictions['House 101']}")
```

Sample output:

_Predicted price for House 101: $127,285.59_

```python
contribution_scores = realapp.produce_feature_contributions(x_input)
contribution_scores["House 101"]
```

Sample output:

| Feature Name               | Feature Value | Contribution | Average/Mode |
| -------------------------- | ------------- | ------------ | ------------ |
| Lot size in square feet    | 9937          | 1137.73      | 10847.56     |
| Original construction date | 1965          | -3514.96     | 1981         |
| ...                        | ...           | ...          | ...          |

```python
from pyreal.visualize import feature_bar_plot

feature_bar_plot(contribution_scores["House 101"])
```

Sample output:

<figure><img src="../../.gitbook/assets/contributions.png" alt=""><figcaption></figcaption></figure>
