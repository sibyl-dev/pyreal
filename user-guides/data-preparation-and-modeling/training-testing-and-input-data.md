# Training, Testing, and Input Data

{% hint style="info" %}
The full code for this and all other user guides can be found in our [user guide tutorial](https://github.com/sibyl-dev/pyreal/blob/dev/tutorials/user\_guide.ipynb).
{% endhint %}

The first step to getting started with Pyreal is to prepare your data.

Pyreal expects data in the format of [Pandas DataFrames](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). Each row refers to one _data instance_ (a person, place, thing, or entity), and each column refers to a _feature_, or piece of information about that instance. Column headers are the names of feature. Each instance may optionally have an instance ID, which can either be stored as the DataFrame's indices (row IDs) or as a separate column.

For example, a part of your data may look like:

| House ID | HouseSize | Neighborhood | BasementSize |
| -------- | --------- | ------------ | ------------ |
| 110      | 1774      | Old Town     | 952          |
| 111      | 1077      | Brookside    | _N/A_        |
| 112      | 1040      | Sawyer       | 1040         |

There are three categories of data relevant to ML decision-making: training data, testing data, and input data.

The training data is used to train the ML model and explainers. The testing data is used to evaluate the performance of the ML model (ie., how accurately it makes predictions). The input data is the data that you actively wish to get predictions on and understand better.&#x20;

For training and test data, we will usually have the _ground truth_ values (the "correct" answer for the value your model tries to predict, often referred to as _y-values_) for all rows of data.&#x20;

For example, if we are trying to predict house prices, you would have additional information about the price of houses in your training/testing datasets. Pyreal expects these target values as [pandas Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html).

<table data-full-width="false"><thead><tr><th data-type="number">house_id</th><th data-type="number">SalePrice ($)</th></tr></thead><tbody><tr><td>110</td><td>129900</td></tr><tr><td>111</td><td>118000</td></tr><tr><td>112</td><td>129500</td></tr></tbody></table>

For the input data, we do not know the ground-truth, we we use the ML model to get a prediction.

### Sample Code

For this user guide, our examples will use a smaller version of the [Ames Housing Dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset), with just 8 key features. You can load in sample data using the Pyreal `sample_applications` module, and use the `train_test_split` function from sklearn to split your data into training and testing sets.

```python
from pyreal.sample_applications import ames_housing_small
from sklearn.model_selection import train_test_split

x, y = ames_housing_small.load_data(include_targets=True)

# x_train and x_test have corresponding ground-truth values in y_train and y_test
x_train, x_test, y_train, y_test = train_test_split(x, y)

# We do not have the ground-truth values for x_input
x_input = ames_housing_small.load_input_data()
```
