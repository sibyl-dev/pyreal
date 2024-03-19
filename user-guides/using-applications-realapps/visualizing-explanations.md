---
description: You can use Pyreal's `visualize` module to quickly get explanation graphs
---

# Visualization

{% hint style="info" %}
The full code for this and all other user guides can be found in our [user guide tutorial](https://github.com/sibyl-dev/pyreal/blob/dev/tutorials/user\_guide.ipynb).
{% endhint %}

Pyreal's visualize module includes several functions that take in RealApp output directly to generate explanation plots.

{% hint style="info" %}
All visualization functions take in many customization parameters. See the[ API reference](https://sibyl-ml.dev/pyreal/api\_reference/visualize.html) for more information.
{% endhint %}

## Feature Bar Plot

The feature bar plot can visualize general feature importance scores...

```python
from pyreal.visualize import feature_bar_plot

feature_bar_plot(importance_scores)

feature_bar_plot(contributions["House 201"]
```

<figure><img src="../../.gitbook/assets/importance.png" alt=""><figcaption><p>In feature bar plots, each bar represents the importance or contribution of the feature. By default, feature bar plots show just the top 5 more important features, but this can be changed with the num_features parameter.</p></figcaption></figure>

... or contribution scores for a single input

```python
feature_bar_plot(contribution_scores["House 101"])
```

<figure><img src="../../.gitbook/assets/contributions.png" alt=""><figcaption><p>For feature contribution bar plots, the contribution from each feature can be negative or positive. The x-axis here represents the total contribution of each feature to the model's prediction, in dollars. The value of each feature can be seen in parenthesis after the y-axis labels.</p></figcaption></figure>

## Strip Plot

Strip plots are an effective way to visualize feature contributions for multiple inputs at a time, to understand the general trends of how the ML model uses features.

To increase the amount of information displayed in these plots, you can generate feature contributions for the full training set.

```python
from pyreal.visualize import strip_plot

training_set_contributions = realapp.produce_feature_contributions(x_train)
strip_plot(training_set_contributions)
```

<figure><img src="../../.gitbook/assets/strip_plot.png" alt=""><figcaption><p>With strip plots, each point represents one house in the dataset. Its color represents the feature value, and its location on the x-axis represents the contribution to the model prediction, in dollars, for that feature on that instance. We can see in the first row that higher qualities tend to increase the model's prediction more, and lower qualities tend to decrease the predicted price (as exptected). Similarly, in the second row we can see the same relationship for house size.</p></figcaption></figure>

## Feature Scatter Plot

Scatter plots allow you to investigate how the model uses a specific feature, across the full range of that feature's values:

```python
from pyreal.visualize import feature_scatter_plot

# Optionally pass in predictions to color the plot by prediction
predictions = realapp.predict(x_train, format=False)

feature_scatter_plot(training_set_contributions, 
                     "Total above ground living area in square feet", 
                     predictions=predictions)
```

<figure><img src="../../.gitbook/assets/scatter_plot.png" alt=""><figcaption><p>In feature scatter plots, each point once again represents one row in the training dataset. The x-axis holds the features values, and the y-axis holds the contributions to the model's predicted price. Again, we can see that larger houses tend to increase the model predictions more. The colors of the dots additionally show the model's overall prediction, considering all features.</p></figcaption></figure>

## Example Table

To get a clean table comparing the feature values of a input data to those of similar examples, you can use the `example_table` function:

```python
from pyreal.visualize import example_table

example_table(similar_houses["House 101"])
```

This will give you a table like:

|                | Ground Truth | Lot size in square feet | Neighborhood | ... |
| -------------- | ------------ | ----------------------- | ------------ | --- |
| Original Input | N/A          | 9937                    | Edwards      | ... |
| House 984      | $144,500.00  | 8562.0                  | Edwards      | ... |
| House 1004     | $149,900.00  | 17755.0                 | Edwards      | ... |

