<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
<!--[![PyPI Shield](https://img.shields.io/pypi/v/explanation-toolkit.svg)](https://pypi.python.org/pypi/explanation-toolkit)-->
<!--[![Downloads](https://pepy.tech/badge/explanation-toolkit)](https://pepy.tech/project/explanation-toolkit)-->
[![Travis CI Shield](https://travis-ci.org/DAI-Lab/explanation-toolkit.svg?branch=master)](https://travis-ci.org/DAI-Lab/explanation-toolkit)
[![Coverage Status](https://codecov.io/gh/DAI-Lab/explanation-toolkit/branch/master/graph/badge.svg)](https://codecov.io/gh/DAI-Lab/explanation-toolkit)

# Explanation Toolkit

Library for evaluating and deploying machine learning explanations.

- Free software: Not open source
- Documentation: https://DAI-Lab.github.io/explanation-toolkit
- Homepage: https://github.com/DAI-Lab/explanation-toolkit

# Overview

TODO: Provide a short overview of the project here.

# Install

## Requirements

**Explanation Toolkit** has been developed and tested on [Python3.4, 3.5, 3.6 and 3.7](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system
in which **Explanation Toolkit** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **Explanation Toolkit**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) explanation-toolkit-venv
```

Afterwards, you have to execute this command to activate the virtualenv:

```bash
source explanation-toolkit-venv/bin/activate
```

Remember to execute it every time you start a new console to work on **Explanation Toolkit**!

<!-- Uncomment this section after releasing the package to PyPI for installation instructions
## Install from PyPI

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **Explanation Toolkit**:

```bash
pip install explanation-toolkit
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).
-->

## Install from source

With your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:DAI-Lab/explanation-toolkit.git
cd explanation-toolkit
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://DAI-Lab.github.io/explanation-toolkit/contributing.html#get-started)
for more details about this process.

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **Explanation Toolkit**.

TODO: Create a step by step guide here.

# What's next?

For more details about **Explanation Toolkit** and all its possibilities
and features, please check the [documentation site](
https://DAI-Lab.github.io/explanation-toolkit/).
