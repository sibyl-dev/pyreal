---
description: >-
  RealApps are encapsulated model objects that allow you to make predictions and
  get explanations for your ML models
---

# Creating new applications (RealApps)

{% hint style="info" %}
The full code for this and all other user guides can be found in our [user guide tutorial](https://github.com/sibyl-dev/pyreal/blob/dev/tutorials/user\_guide.ipynb).
{% endhint %}

In the [Data Preparation and Modeling guide](data-preparation-and-modelling/), you learned how to set up your data, transformers, and model for an ML application.&#x20;

In this guide, you will learn how to combine these components into a RealApp object, which in turn makes it fast and easy to use and understand your ML models.

## Initializing RealApps

At minimum, RealApps require the ML model and any transformers necessary to prepare input data for model predictions. For example, following up our example from the [data preparation](data-preparation-and-modelling/) guide:

<pre class="language-python"><code class="lang-python">from pyreal import RealApp

<strong>realapp = RealApp(model,
</strong>                  X_train_orig=x_train,
                  y_train=y_train, 
                  transformers=transformers)
</code></pre>

{% hint style="info" %}
RealApps take in a parameter called `X_train_orig`. You can read this as "training data in the original feature space". Data in the _original feature space_ is data in whatever format the transformer list operates on. For example, in our sample dataset from the [data preparation guide](data-preparation-and-modelling/), we started with data that had categorical, unstandardized features with missing data. Our transformer list then prepares this data for the model.
{% endhint %}

## Setting ID Columns

RealApp predictions and explanations are indexed by instance ID. By default, these IDs are taken from the input data's index (row labels in pandas DataFrames). However, you can set a different column to be the ID column instead.&#x20;

The ID column will be automatically dropped before running the data through the ML model.

```python
from pyreal import RealApp

realapp = RealApp(model,
                  X_train_orig=x_train,
                  y_train=y_train, 
                  transformers=transformers,
                  id_column="House ID")
```

## Adding Feature Descriptions

In some datasets, the names given to features (the column names in your DataFrame) may not be very readable. Keeping simpler column names is useful for tasks such as assigning transformers to features, but for the final information and explanations presented to users you may prefer more readable descriptions.

In this case, you can additionally pass a `feature_descriptions` dictionary.

```python
from pyreal import RealApp

feature_descriptions = {'HouseSize': 'Total above ground living area in square feet',
                        'Material': 'Exterior material of house',
                        ...}

realapp = RealApp(model,
                  X_train_orig=x_train, 
                  y_train=y_train,
                  transformers=transformers,
                  feature_descriptions=feature_descriptions)
```

## Formatting Predictions

You can format predictions coming from RealApp objects (such as setting decimal places for numeric predictions, or converting True/False predictions to more meaningful values) with the `pred_format_func` parameter

<pre class="language-python"><code class="lang-python"><strong>from pyreal import RealApp
</strong>
# Format numerics into dollar amounts (1523.273 -> $1,523.27)
realapp = RealApp(model,
                  X_train_orig=x_train, 
                  y_train=y_train,
                  transformers=transformers,
                  pred_format_func=lambda x: f"${x:,.2f}")
              
# Format True/False into Failure/Success
def format_func(pred):
     return "Failure" if x else "Success"
     
realapp = RealApp(model,
                  X_train_orig=x_train, 
                  y_train=y_train,
                  transformers=transformers,
                  pred_format_func=format_func)
</code></pre>

## Passing Data at Fit Time

Passing training data to your RealApp at initialization time allows the app to automatically access this data any time it needs to for fitting explainers, which allows for convenient and easy coding. However, storing this data may greatly increase the size of the RealApp object if you dataset is large.

If you need smaller RealApp objects, or would otherwise like to avoid storing the data with the object, you can avoid passing data at initialization and instead pass it manually any time you are preparing a new explainer (see the [Using Applications](generating-explanations/) guide for more details on this process)

```python
from pyreal import RealApp

realapp = RealApp(model,
                  transformers=transformers)

# You will need to manually pass data in when preparing explainers.
app.prepare_feature_contributions(x_train, y_train)
```
