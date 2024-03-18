---
description: You can use Pyreal's `visualize` module to quickly get explanation graphs
---

# Visualization

Pyreal's visualize module includes several functions that take in RealApp output directly to generate explanation plots.

{% hint style="info" %}
All visualization functions take in many customization parameters. See the[ API reference](https://sibyl-ml.dev/pyreal/api\_reference/visualize.html) for more information.
{% endhint %}

## Feature Bar Plot

The feature bar plot can visualize general feature importance or feature contributions for a single input.

```python
from pyreal.visualize import feature_bar_plot

# create or load realapp and input data (houses) as in previous guides

importance = realapp.produce_feature_importance()
feature_bar_plot(importance)

contributions = realapp.produce_feature_contributions(houses)
feature_bar_plot(contributions["House 201"]
```

## Strip Plot

Strip plots are an effective way to visualize feature contributions for multiple inputs at a time, to understand the general trends of how the ML model uses features.

```python
from pyreal.visualize import strip_plot

contributions = realapp.produce_feature_contributions(houses)
strip_plot(contributions)
```

## Feature Scatter Plot

Scatter plots allow you to investigate how the model uses a specific feature, across the full range of that feature's values:

```python
from pyreal.visualize import feature_scatter_plot

contributions = realapp.produce_feature_contributions(houses)

# Optionally pass in predictions to color the plot by prediction
predictions = realapp.predict(houses)

feature_scatter_plot(contributions, 
                     feature="Lot size in square feet",
                     predictions=predictions)
```

## Example Table

To get a clean table comparing the feature values of a input data to those of similiar examples, you can use the `example_table` function:

```python
from pyreal.visualize import example_table

similar_houses = realapp.produce_similar_examples(houses)

example_table(similiar_houses)
```
