---
description: >-
  Pyreal makes it easy to understand your ML model and its predictions,
  generating predictions in a readable format.
---

# Explanations

{% hint style="info" %}
The full code for this and all other user guides can be found in our [user guide tutorial](https://github.com/sibyl-dev/pyreal/blob/dev/tutorials/user\_guide.ipynb).
{% endhint %}

Sometimes you may want more information about how an ML model came up with its prediction on an input. Or, you may have questions about the ML model in general. For our example application of predicting house prices based on information about houses, you may have questions like:

1. [What information about the house (or _features_) contributed to the prediction of $102,300?](explanations.md#what-features-contributed-to-the-model-prediction)
2. [Have we seen houses in the past similar to this one, and what were their prices?](explanations.md#which-past-cases-are-similiar-to-this-one)
3. [What features, in general, does the model consider most important for predicting house prices?](explanations.md#what-features-does-the-model-consider-most-important-in-general)
4. [How does the model use the "size" feature? Are bigger houses always predicted to be more expensive?](explanations.md#how-does-the-model-use-a-specific-feature)

You can answer these questions by look at ML model explanations. Explanations are generated using the `RealApp` class's `.produce()` functions using the name of the explanation type you desire.

Pyreal generates explanations that are naturally readable, in the same _feature space_ as your input data. This means explanations will show the original values of features, before scaling, one-hot encoding, imputing, or any other muddling transformations.&#x20;

{% hint style="info" %}
To make Pyreal's explanations even more readable, you can use advanced transformers, as described in the [Transformers: Extended Guide](../data-preparation-and-modeling/transformers/transformers-extended-guide.md).
{% endhint %}

## Sample Explanations

In this guide, we will go through a few common questions you may have your model/model predictions, and the appropriate Pyreal explanation functions to answer them.

### What features contributed to the model prediction?

To get a list of how much each feature in your input data contributed to the model's prediction, you can use the `.produce_feature_contributions(x_orig)` function

```python
contribution_scores = realapp.produce_feature_contributions(x_input)
```

{% hint style="info" %}
You can read `x_orig` as the input data in the original feature space. That means, you pass your data into Pyreal in its original format, without any transformations. The transformers passed to your RealApp object handle the rest.
{% endhint %}

Feature contribution outputs from RealApps are indexed by row ids, found in the column labelled by the optional `id_column` parameter to RealApps, or by the index (row labels) of the input data if no id column is provided.&#x20;

This allows us to access the explanation for a given house by ID:

```python
contributions ["House 101"]
```

... which outputs a DataFrame with all features, their contributions, and their average or mode value:

| Feature Name               | Feature Value | Contribution | Average/Mode |
| -------------------------- | ------------- | ------------ | ------------ |
| Lot size in square feet    | 9937          | 1137.73      | 10847.56     |
| Original construction date | 1965          | -3514.96     | 1981         |
| ...                        | ...           | ...          | ...          |

The default algorithm for computing feature contributions is SHAP, which means the contribution values take the same units as the model's prediction. For example, in this case the lot size increased the predicted price of the house by $1,137.73.

If you are only interested in the most contributing features (either positively, negatively, or by absolute value), you can using the `num_features` and `select_by` parameters. Alternatively, you can extract the top contributing features from an already-generated explanation using the `get_top_contributors` function on the contribution explanation for your input of interest.

```python
from pyreal.utils import get_top_contributors

# select_by is one of: "absolute", "min", "max"
top_contributions_for_house_101 = get_top_contributors(contribution_scores["House 101"], 
                                                       num_features=5, 
                                                       select_by="absolute")

# Or...
top_5_contribution_scores = realapp.produce_feature_contributions(x_input, 
                                                                  num_features=5,
                                                                  select_by="absolute")
top_5_contribution_scores["House 101"]
```

### Which past cases are similar to this one?

You can get a list of past cases (rows of data in the training data) that are similar to your input data, as well as the ground-truth target (y) value for those cases, by using the `produce_similar_examples` function:

```python
# Get the three most similar houses from the 
# training dataset to each house in houses
similar_houses = realapp.produce_similar_examples(x_input, 
                                                  num_examples=3)
```

The return type for these explanations is a dictionary indexed by row IDs (either from the specified ID column or the index (row labels) of the input data). For each ID, `similar_houses[ID]` contains:

* `similar_houses[ID][X]` (_DataFrame):_ The feature values of the houses most similar to the house specified by ID.
* `similar_houses[ID][y]` (_Series_): The corresponding ground-truth target values (y) for all similar houses.
* `similar_houses[ID][Input]` (_Series_): The input features (ie. the feature values for the house specified by ID) in the same feature space as the similar examples

{% hint style="info" %}
The output for similar examples includes the input values because the transformers you pass in may result in outputs in a different feature space than the one you pass your data in with. This simply ensures you have the same feature space available for both.
{% endhint %}

### What features does the model consider most important in general?

You may be interested in understanding which features the model considers most important in general, without considering a specific input. For this, you can use the `produce_feature_importance` function, which takes no required inputs:

<pre class="language-python"><code class="lang-python"><strong>importance_scores = realapp.produce_feature_importance()
</strong><strong>
</strong><strong># Like with feature contributions, you can return only the most important features
</strong><strong>#  or extract the most important features using `get_top_contributors`
</strong>top_5_importance_scores = realapp.produce_feature_importance(num_features=5)
</code></pre>

... which generates a DataFrame of feature importance values

| Feature Name                                                | Importance |
| ----------------------------------------------------------- | ---------- |
| Overall quality of the house finishing and materials (1-10) | 23149.06   |
| Total above ground living area in square feet               | 17930.23   |
| ...                                                         | ...        |

{% hint style="info" %}
These importance scores are unitless, meaning they should only be considered in relative terms to other importance score.
{% endhint %}

### How does the model use a specific feature?

To understand how the model uses a specific feature, you can generate feature contributions for the full training dataset, and then investigate how the contributions vary for a specific feature by value.

To save time generating large numbers of contributions, and get the output in a more usable format for this specific use-case, you can set the `format_output` parameter to produce functions to `False`.

```python
contribution_scores_df = realapp.produce_feature_contributions(
    x_input, format_output=False)

# explanations is now a tuple of (feature contributions, feature values), where
#  the column names are feature descriptions and the index are the row ids.
contributions, values = contribution_scores_df 

# You can now investigate a single features contributions with:
contributions["Lot size in square feet"]
values["Lot size in square feet"]
```

{% hint style="info" %}
The easiest way to understand a feature's contribution across a dataset is with a plot. See the [visualization ](visualizing-explanations.md)guide for code to do this with Pyreal.
{% endhint %}

## Next steps

Pyreal outputs are formatted to be easily usable with whatever method you find more natural. The next few guides go through a few built-in options for better displaying explanations.
