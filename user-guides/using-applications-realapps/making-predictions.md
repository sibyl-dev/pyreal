# Making Predictions

## Making Predictions

The most basic yet important functionality of an ML model is making predictions. For this, you can use the `.predict()` function. This function takes data in the original format, and then runs all transformers needed to prepare your input data for the model.

```python
import pandas as pd
from pyreal import RealApp

realapp = RealApp(model,
                  X_train_orig=X_train, 
                  y_train=y_train,
                  transformers=transformers,
                  pred_format_func='${:,.2f}'.format)

X_input = pd.DataFrame([[2300, "Pinewood", 50],
                        [1850, "Placedale", None]],
                        columns=["size", "location", "garden_size"])
realapp.predict(X_input)
# Sample output: {0: "$102,300", 1: "$342,000"}
```

The output of the predict function is a dictionary with keys determined by input index (row names) by default. Alternatively, the keys can be taken from an ID column if specified:

```
import pandas as pd
from pyreal import RealApp

realapp = RealApp(model,
                  X_train_orig=X_train, 
                  y_train=y_train,
                  transformers=transformers,
                  pred_format_func='${:,.2f}'.format,
                  id_column="house_id")

X_input = pd.DataFrame([["house 201", 2300, "Pinewood", 50],
                        ["house 202", 1850, "Placedale", None]],
                        columns=["house_id", "size", "location", "garden_size"])
realapp.predict(X_input)
# Sample output: {"house 201": "$102,300", "house 202": "$342,000"}
```

{% hint style="info" %}
To avoid formatting the output as a dictionary, and instead just get a list of predictions, you can use the `format` parameter: `realap.predict(X, format=False)`
{% endhint %}
