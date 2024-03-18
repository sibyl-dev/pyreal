# Training and Input Data

The first step to getting started with Pyreal is to prepare your data.

Pyreal expects data in the format of [Pandas DataFrames](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). Each row refers to one _data instance_ (a person, place, thing, or entity), and each column refers to a _feature_, or piece of information about that instance. Column headers are the names of feature. Each instance may optionally have an instance ID, which can either be stored as the DataFrame's indices (row IDs) or as a separate column.

For example, a part of your data may look like:

<table><thead><tr><th data-type="number">house_id</th><th data-type="number">size</th><th>location</th><th>garden_size</th></tr></thead><tbody><tr><td>101</td><td>2200</td><td>Pinewood</td><td>100</td></tr><tr><td>102</td><td>1500</td><td>Oceanview</td><td><em>N/A</em></td></tr><tr><td>103</td><td>1800</td><td>Placedale</td><td>120</td></tr></tbody></table>

There are two categories of data relevant to ML decision-making: the training data and the input data.

The training data is used to train the ML model and explainers. The input data is the data that you actively wish to get predictions on and understand better. The main difference between these two types is data is that you usually will have the _ground truth_ values (the "correct" answer for the value your model tries to predict) for your training data but not your input data.

For example, if we are trying to predict house prices, you would have additional information about the price of houses in your training dataset.

<table data-full-width="false"><thead><tr><th data-type="number">house_id</th><th data-type="number">price ($)</th></tr></thead><tbody><tr><td>101</td><td>250000</td></tr><tr><td>102</td><td>220000</td></tr><tr><td>103</td><td>180000</td></tr></tbody></table>
