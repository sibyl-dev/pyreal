.. _quickstart:

Quickstart
==========

In this short tutorial we will guide you through a series of steps that will help you
getting started with **Pyreal**.

Get a Feature Contribution Explanation
--------------------------------------
To get a feature contribution explanation of a pre-trained model, we can use the
``LocalFeatureContribution`` class. We begin by loading in the model and transformers.

.. ipython:: python
    :okwarning:

    import pyreal.applications.titanic as titanic
    from pyreal.transformers import ColumnDropTransformer, MultiTypeImputer

    # Load in data
    x_orig, y = titanic.load_titanic_data()

    # Load in feature descriptions -> dict(feature_name: feature_description, ...)
    feature_descriptions = titanic.load_feature_descriptions()

    # Load in model
    model = titanic.load_titanic_model()

    # Load in list of transformers
    transformers = titanic.load_titanic_transformers()

The ``transformers`` object is a list that includes three types of transformers, specific for this
application:

- ``ColumnDropTransformer``: removes features that should not be used in prediction
- ``MultiTypeImputer``: replaces missing data from all columns with a reasonable replacement
- ``OneHotEncoderWrapper``: one-hot encodes categorical features. We use the built-in wrapper type,
  which includes a ``transform_explanation`` function.

These transformers transform the data from it's `original` state (``x_orig``) to its
`explanation ready` state (``x_explain``). In this case, the explanation algorithm used expects
data in the `model ready` state (``x_model``), so ``x_explain == x_model``.

Next, we can create the ``Explainer`` object, and fit it.

.. ipython:: python
    :okwarning:

    from pyreal.explainers import LocalFeatureContribution
    lfc = LocalFeatureContribution(model=model, x_train_orig=x_orig,
                                   m_transforms=transformers, e_transforms=transformers,
                                   feature_descriptions=feature_descriptions, fit_on_init=True)
    lfc.fit()

Finally, we can get the explanation using the ``.produce()`` function. We will also visualize
the most contributing features using the `visualize` model.

.. ipython:: python
    :okwarning:

    input_to_explain = x_orig.iloc[0]
    prediction = lfc.model_predict(input_to_explain) # Prediction: [0]

    contributions = lfc.produce(input_to_explain)

    from pyreal.utils import visualize
    x_interpret = lfc.convert_data_to_interpretable(input_to_explain)

    # Plot a bar plot of top contributing features, by asbolute value
    visualize.plot_top_contributors(contributions, select_by="absolute", values=x_interpret)

The output will be a bar plot showing the most contributing features, by absolute value.

.. image:: /images/quickstart.png

We can see here that the input passenger's predicted chance of survival was greatly reduced
because of their sex (male) and ticket class (3rd class).
