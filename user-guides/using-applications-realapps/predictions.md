---
description: >-
  You can use RealApps to make predictions with your ML model. All data
  transformations required and handled under-the-hood.
---

# Predictions

{% hint style="info" %}
The full code for this and all other user guides can be found in our [user guide tutorial](https://github.com/sibyl-dev/pyreal/blob/dev/tutorials/user\_guide.ipynb).
{% endhint %}

The most basic yet important functionality of an ML model is making predictions. For this, you can use the `.predict()` function. This function takes data in the original format, and then runs all transformers needed to prepare your input data for the model.

```python
from pyreal import RealApp

predictions = realapp.predict(x_input)
# predictions: {"House 101": "$149,003.42", "House 102": "$188,169.59", ...}
```

The output of the predict function is a dictionary with keys determined by the ID column. If not ID column is provided, the output is indexed by input index (row names). This allows you to access predictions for specific instances by ID.

```python
predictions["House 101"]
# Output: $149,003.42
```

{% hint style="info" %}
To avoid formatting the output as a dictionary, and instead just get a list of predictions, you can use the `format` parameter: `realap.predict(X, format=False)`
{% endhint %}
