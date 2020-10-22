<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
<!--[![PyPI Shield](https://img.shields.io/pypi/v/pyreal.svg)](https://pypi.python.org/pypi/pyreal)-->
<!--[![Downloads](https://pepy.tech/badge/pyreal)](https://pepy.tech/project/pyreal)-->
[![Travis CI Shield](https://travis-ci.org/DAI-Lab/pyreal.svg?branch=master)](https://travis-ci.org/DAI-Lab/pyreal)
[![Coverage Status](https://codecov.io/gh/DAI-Lab/pyreal/branch/master/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/pyreal)

# Pyreal

Library for evaluating and deploying machine learning explanations.

- Free software: Not open source
- Documentation: https://DAI-Lab.github.io/pyreal
- Homepage: https://github.com/DAI-Lab/pyreal

# Overview

**Pyreal** wraps the complete machine learning explainability pipeline into Explainer objects. Explainer objects
handle all the transforming logic, in order to provide a human-interpretable explanation from any original
data form.

# Install

## Requirements

**Pyreal** has been developed and tested on [Python3.4, 3.5, 3.6 and 3.7](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system
in which **Pyreal** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **Pyreal**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) pyreal-venv
```

Afterwards, you have to execute this command to activate the virtualenv:

```bash
source pyreal-venv/bin/activate
```

Remember to execute it every time you start a new console to work on **Pyreal**!

<!-- Uncomment this section after releasing the package to PyPI for installation instructions
## Install from PyPI

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **Pyreal**:

```bash
pip install pyreal
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).
-->

## Install from source

With your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:DAI-Lab/pyreal.git
cd pyreal
git checkout stable
make install
```

<!--## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://DAI-Lab.github.io/pyreal/contributing.html#get-started)
for more details about this process.-->

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **Pyreal**. For a more detailed version of this tutorial, see 
`examples.titanic.titanic_tutorial.ipynb`

```python3
from pyreal.explainers import LocalFeatureContribution
import pyreal.applications.titanic as titanic
from pyreal.utils.transformer import ColumnDropTransformer, MultiTypeImputer
from pyreal.utils import visualize

# First, we will load in the Titanic dataset
x_orig, y = titanic.load_titanic_data()

# Next, we load in a dictionary that provides human-readable descriptions of the feature names
#   Format: {feature_name : feature_description, ...}
feature_descriptions = titanic.load_feature_descriptions()

# Finally, we load in the trained model and corresponding fitted transformers
model = titanic.load_titanic_model()
transformers = titanic.load_titanic_transformers()

# Now, we can make and fit a LocalFeatureContribution object, which will handle all the 
#   transformations needed to get an interpretable SHAP feature contribution explanation
lfc = LocalFeatureContribution(model=model, x_orig=x_orig, m_transforms=transformers, e_transforms=transformers, 
                               contribution_transforms=transformers, 
                               feature_descriptions=feature_descriptions)
lfc.fit()

# We can now choose an input, and see the model's prediction.
input_to_explain = x_orig.iloc[0]
print("Prediction:", lfc.model_predict(input_to_explain)) # Output -> Prediction: [0]

# We see that this person is not predicted to survive. 
#   Let's see why, by using LocalFeatureContribution's .produce() function
contributions = lfc.produce(input_to_explain)

# We can visualize the most contributing features using the pyreal.utils.visualize module. 
#   We will also convert our input to the interpretable space, so we can add it's values to
#   the visualization
x_interpret = lfc.convert_data_to_interpretable(input_to_explain)
visualize.plot_top_contributors(contributions, select_by="absolute", values=x_interpret)
```
The output will be a bar plot showing the most contributing features, by absolute value. 

![Quickstart](docs/images/quickstart.png)

We can see here that the input passenger's predicted chance of survival was greatly reduced
because of their sex (male) and ticket class (3rd class).

# What's next?

For more details about **Pyreal** and all its possibilities
and features, please check the [documentation site](
https://DAI-Lab.github.io/pyreal/).
