---
description: Introduction to preparing the components of ML decision-making
---

# Data Preparation and Modelling

{% hint style="info" %}
The full code for this and all other user guides can be found in our user guide tutorial.
{% endhint %}

Pyreal wraps all the components you need for an  ML decision-making use-case in a single RealApp object. This allows you to make model predictions, transform data, get explanations, and otherwise interact with your ML model for decision-making.&#x20;

There are three key components:

1. The data
2. Transformers that transform the data for the ML model
3. The ML model

In this guide, we go through these components and introduce the process of preparing them.&#x20;

{% hint style="info" %}
If you already have a fully set-up ML workflow (including an ML model, data, and possibly data transformers), you can head over to the [Migrating to Pyreal](https://github.com/sibyl-dev/pyreal/blob/dev/tutorials/migrating\_to\_pyreal.ipynb) tutorial to get started.
{% endhint %}

## In this guide...

<table data-view="cards"><thead><tr><th></th><th></th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td><h3><a href="./#training-and-input-data-1">Training, Testing, and Input Data</a></h3></td><td>Learn to prepare your data as pandas DataFrames</td><td><a href="training-and-input-data.md">training-and-input-data.md</a></td></tr><tr><td><h3><a href="./#transformers-1">Transformers</a></h3></td><td>Learn to create Pyreal data transformers for feature engineering </td><td><a href="transformers/">transformers</a></td></tr><tr><td><h3><a href="modelling.md">Modelling</a></h3></td><td>Learn how to train models to make predictions on your data</td><td><a href="modelling.md">modelling.md</a></td></tr></tbody></table>

