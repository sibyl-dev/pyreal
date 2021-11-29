<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
[![PyPI Shield](https://img.shields.io/pypi/v/pyreal.svg)](https://pypi.python.org/pypi/pyreal)
<!--[![Downloads](https://pepy.tech/badge/pyreal)](https://pepy.tech/project/pyreal)-->
<!--[![Travis CI Shield](https://travis-ci.org/DAI-Lab/pyreal.svg?branch=master)](https://travis-ci.org/DAI-Lab/pyreal)-->
<!--[![Coverage Status](https://codecov.io/gh/DAI-Lab/pyreal/branch/master/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/pyreal)-->
[![Build Action Status](https://github.com/DAI-Lab/pyreal/workflows/Test%20CI/badge.svg)](https://github.com/DAI-Lab/pyreal/actions)
# Pyreal

Library for evaluating and deploying machine learning explanations.

- Free software: Not open source
- Documentation: https://sibyl-dev.github.io/pyreal
- Homepage: https://sibyl-ml.dev/

# Overview

**Pyreal** wraps the complete machine learning explainability pipeline into Explainer objects. Explainer objects
handle all the transforming logic, in order to provide a human-interpretable explanation from any original
data form.

# Install

## Requirements

**Pyreal** has been developed and tested on [Python 3.7, 3.8, and 3.9](https://www.python.org/downloads/)
The library uses Poetry for package management.

## Install from PyPI

We recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **Pyreal**:

```
pip install pyreal
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).

## Install from source
If you do not have **poetry** installed, please head to [poetry installation guide](https://python-poetry.org/docs/#installation)
and install poetry according to the instructions.\
Run the following command to make sure poetry is activated. You may need to close and reopen the terminal.

```
poetry --version
```

Finally, you can clone this repository and install it from
source by running `poetry install`:

```
git clone git@github.com:DAI-Lab/pyreal.git
cd pyreal
poetry install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://sibyl-dev.github.io/pyreal/developer_guides/contributing.html)
for more details about this process.

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **Pyreal**. We will get an explanation for a prediction on whether a
passenger on the Titanic would have survived.

 For a more detailed version of this tutorial, see
`examples.titanic.titanic_lfc.ipynb`

#### Load in demo dataset, pre-fit model, and transformers
```
>>> import pyreal.applications.titanic as titanic
>>> from pyreal.transformers import ColumnDropTransformer, MultiTypeImputer

# Load in data
>>> x_train_orig, y = titanic.load_titanic_data()

# Load in feature descriptions -> dict(feature_name: feature_description, ...)
>>> feature_descriptions = titanic.load_feature_descriptions()

# Load in model
>>> model = titanic.load_titanic_model()

# Load in list of transformers
>>> transformers = titanic.load_titanic_transformers()

# Create and fit LocalFeatureContribution Explainer object
>>> from pyreal.explainers import LocalFeatureContribution
>>> lfc = LocalFeatureContribution(model=model, x_train_orig=x_train_orig,
...                                transformers=transformers,
...                                feature_descriptions=feature_descriptions,
...                                fit_on_init=True)
>>> lfc.fit()

# Make predictions on an input
>>> input_to_explain = x_train_orig.iloc[0]
>>> prediction = lfc.model_predict(input_to_explain) # Prediction: [0]

# Explain an input
>>> contributions = lfc.produce(input_to_explain)

# Visualize the explanation
>>> from pyreal.utils import visualize
>>> x_interpret = lfc.convert_data_to_interpretable(input_to_explain)

```

<!--## Install for Development

TODO: Running tests should not bring up a window. Refactor into the above docstring, not actually spawning the subsequent window-->

##### Plot a bar plot of top contributing features, by absolute value
```
visualize.plot_top_contributors(contributions, select_by="absolute", values=x_interpret)
```


The output will be a bar plot showing the most contributing features, by absolute value.

![Quickstart](docs/images/quickstart.png)

We can see here that the input passenger's predicted chance of survival was greatly reduced
because of their sex (male) and ticket class (3rd class).

### Terminology
Pyreal introduces specific terms and naming schemes to refer to different feature spaces and
transformations. Here, we offer a short summary of these terms. You can see examples of
some of these in the quick start tutorial above.

| Term            | Description             |
|-----------------|-------------------------|
| `model`         | A machine learning predictor, defined as an object with a `.predict()` funtion     |
| `x`, `x_train`  | Data, in the form of a pandas DataFrame            |
| `explanation`   | An explanation of a model or model prediction             |
| `Explainer`     | Pyreal objects that take in data and a model and return an explanation |
| `Transformer`   | Pyreal objects that transform data and explanations from one feature space to another |
| `x_orig`, `explanation_orig` | Data or an explanation using the original feature space (whatever feature space the data starts in) |
| `x_explain`, `explanation_explain` | Data or an explanation using the feature space expected by the explanation algorithm (explanation-ready feature space) |
| `x_model`, `explanation_model` | Data or an explanation using the feature space the model expects
| `x_interpret`, `explanation_interpret` | Data or an explanation using the most human-readable feature space available |
| `e_transformers` | Transformers that transform data from the original to the explanation-ready, and explanations from the explanation-ready to the original feature space |
| `m_transformers` | Transformers that transform data between the explanation-ready and model-ready feature spaces |
| `i_transformers` | Transformers that transform data and explanations from the original feature space to the interpretable feature space |

For more details about how these feature spaces and terms interact, please check the user guides.

# What's next?

For more details about **Pyreal** and all its possibilities
and features, please check the [documentation site](
https://sibyl-dev.github.io/pyreal/).
