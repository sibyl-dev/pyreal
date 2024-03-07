<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->

[![PyPI Shield](https://img.shields.io/pypi/v/pyreal.svg)](https://pypi.python.org/pypi/pyreal)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyreal)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pyreal)](https://pypi.python.org/pypi/pyreal)
[![Build Action Status](https://github.com/DAI-Lab/pyreal/workflows/Test%20CI/badge.svg)](https://github.com/DAI-Lab/pyreal/actions)
[![Static Badge](https://img.shields.io/badge/slack-sibyl-purple?logo=slack)](https://join.slack.com/t/sibyl-ml/shared_invite/zt-2dyfwbgo7-2ALinuT2KDZpsVJ4rntJuA)
<!--[![Travis CI Shield](https://travis-ci.org/DAI-Lab/pyreal.svg?branch=stable)](https://travis-ci.org/DAI-Lab/pyreal)-->
<!--[![Coverage Status](https://codecov.io/gh/DAI-Lab/pyreal/branch/stable/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/pyreal)-->

# Pyreal

An easier approach to understanding your model's predictions.

| Important Links                               |                                                                      |
| --------------------------------------------- | -------------------------------------------------------------------- |
| :book: **[Documentation]**                    | Quickstarts and user guides                                          |
| :memo: **[API Reference]**                    | Endpoint usage and details                                           |
| :star: **[Tutorials]**                        | Checkout our notebooks                                               |
| :scroll: **[License]**                        | The repository is published under the MIT License                    |
| :computer: **[Website]**                      | Check out the Sibyl Project Website for more information             |

[Website]: https://sibyl-ml.dev/
[Documentation]: https://dtail.gitbook.io/pyreal/
[Tutorials]: https://github.com/sibyl-dev/pyreal/tree/dev/tutorials
[License]: https://github.com/sibyl-dev/pyreal/blob/dev/LICENSE
[Community]: https://join.slack.com/t/sibyl-ml/shared_invite/zt-2dyfwbgo7-2ALinuT2KDZpsVJ4rntJuA
[API Reference]: https://sibyl-ml.dev/pyreal/api_reference/index.html

# Overview

**Pyreal** gives you easy-to-understand explanations of your machine learning models in a low-code manner.
Pyreal wraps full ML pipelines in a RealApp object that makes it easy to use, understand, and interact with your ML model — regardless of your ML expertise.

# Install

## Requirements

**Pyreal** has been developed and tested on [Python 3.9, 3.10, and 3.11](https://www.python.org/downloads/)
The library uses Poetry for package management.

## Install from PyPI

We recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **Pyreal**:

```
pip install pyreal
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/project/pyreal/).

## Install from source

If you do not have **poetry** installed, please head to [poetry installation guide](https://python-poetry.org/docs/#installation)
and install poetry according to the instructions.\
Run the following command to make sure poetry is activated. You may need to close and reopen the terminal.

```
poetry --version
```

Finally, you can clone this repository and install it from
source by running `poetry install`, with the optional `examples` extras if you'd like to run our tutorial scripts.

```
git clone https://github.com/sibyl-dev/pyreal.git
cd pyreal
poetry install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://dtail.gitbook.io/pyreal/developer-guides/contributing-to-pyreal)
for more details about this process.

# Quickstart

In this short tutorial we will guide you through some steps to get your started with **Pyreal**.
We will use a RealApp object to get predictions and explanations on whether a passenger on the Titanic would have survived.

For a more detailed version of this tutorial, see [our documentation](https://dtail.gitbook.io/pyreal/getting-started/quickstart).

#### Load in the demo data and application

```python
import pyreal.sample_applications.titanic as titanic

real_app = titanic.load_app()
sample_data = titanic.load_data(n_rows=300)

```

#### Predict and produce explanation

```python
predictions = real_app.predict(sample_data)

explanation = real_app.produce_feature_contributions(sample_data)

```

#### Visualize explanation for one passenger

```python
passenger_id = 1
feature_bar_plot(explanation[passenger_id], prediction=predictions[passenger_id], show=False)

```

The output will be a bar plot showing the most contributing features, by absolute value.

![Quickstart](docs/images/titanic.png)

We can see here that the input passenger's predicted chance of survival was greatly reduced
because of their sex (male) and ticket class (3rd class).

### Migrating your application to Pyreal
To create a RealApp object for your own application,
see our [migration tutorial](https://github.com/sibyl-dev/pyreal/blob/dev/tutorials/migrating_to_pyreal.ipynb).

For basic applications built on `sklearn` pipelines, you may be able to simply use:
```python
from pyreal import RealApp

pipeline = # YOUR SKLEARN PIPELINE
X_train, y_train = # YOUR TRAINING DATA

real_app = RealApp.from_sklearn(pipeline, X_train=X_train, y_train=y_train)
```

# Next Steps

For more information on using **Pyreal** for your use case, head over to the full [documentation site](https://dtail.gitbook.io/pyreal/getting-started/next-steps).
