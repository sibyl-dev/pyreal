# Transformers

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

imputer = MultiTypeImputer()
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
