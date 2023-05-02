<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
[![PyPI Shield](https://img.shields.io/pypi/v/pyreal.svg)](https://pypi.python.org/pypi/pyreal)
<!--[![Downloads](https://pepy.tech/badge/pyreal)](https://pepy.tech/project/pyreal)-->
<!--[![Travis CI Shield](https://travis-ci.org/DAI-Lab/pyreal.svg?branch=stable)](https://travis-ci.org/DAI-Lab/pyreal)-->
<!--[![Coverage Status](https://codecov.io/gh/DAI-Lab/pyreal/branch/stable/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/pyreal)-->
[![Build Action Status](https://github.com/DAI-Lab/pyreal/workflows/Test%20CI/badge.svg)](https://github.com/DAI-Lab/pyreal/actions)
# Pyreal

Library for evaluating and deploying machine learning explanations.

- License: MIT
- Documentation: https://pyreal.gitbook.io/pyreal
- API Documentation: https://sibyl-ml.dev/pyreal/api_reference/index.html
- Homepage: https://sibyl-ml.dev/

# Overview

**Pyreal** wraps the complete machine learning explainability pipeline into RealApp objects, which seamlessly
provide usable explanations in a low-code manner.

# Install

## Requirements

**Pyreal** has been developed and tested on [Python 3.8, 3.9, and 3.10](https://www.python.org/downloads/)
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
git clone git@github.com:DAI-Lab/pyreal.git
cd pyreal
poetry install -E examples
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://dtail.gitbook.io/pyreal/developer-guides/contributing-to-pyreal)
for more details about this process.

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **Pyreal**. We will get an explanation for a prediction on whether a
passenger on the Titanic would have survived.

 For a more detailed version of this tutorial, see [our documentation](https://dtail.gitbook.io/pyreal/getting-started/quickstart).

#### Load in the demo data and application
```
>>> import pyreal.sample_applications.titanic as titanic

>>> real_app = titanic.load_app()
>>> sample_data = titanic.load_data(n_rows=300)

```
#### Predict and produce explanation
```
>>> predictions = real_app.predict(sample_data)

>>> explanation = real_app.produce_feature_contributions(sample_data)

```
#### Visualize explanation for one passenger
```
passenger_id = 1
plot_top_contributors(explanation[passenger_id], prediction=predictions[passenger_id], show=False)

```

The output will be a bar plot showing the most contributing features, by absolute value.

![Quickstart](docs/images/titanic.png)

We can see here that the input passenger's predicted chance of survival was greatly reduced
because of their sex (male) and ticket class (3rd class).

### Terminology
Pyreal introduces specific terms and naming schemes to refer to different feature spaces and
transformations. The [Terminology User Guide](https://dtail.gitbook.io/pyreal/developing-applications/developer-terminology-guide) provides an introduction to these terms.

# What's next?

For more details about **Pyreal** and all its possibilities
and features, please check the [documentation site](
https://dtail.gitbook.io/pyreal/).
