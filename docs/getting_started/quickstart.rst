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

The ``transformers`` object is a list that includes three types of transformers, specific for this
application:

- ``ColumnDropTransformer``: removes features that should not be used in prediction
- ``MultiTypeImputer``: replaces missing data from all columns with a reasonable replacement
- ``OneHotEncoderWrapper``: one-hot encodes categorical features. We use the built-in wrapper type,
  which includes a ``transform_contributions`` function.

These transformers transform the data from it's `original` state (``x_orig``) to its
`explanation ready` state (``x_explain``). In this case, the explanation algorithm used expects
data in the `model ready` state (``x_model``), so ``x_explain == x_model``.

Next, we can create the ``Explainer`` object, and fit it.

.. ipython:: python
    :okwarning:

    lfc = LocalFeatureContribution(model=model, x_orig=x_orig, m_transforms=transformers, e_transforms=transformers,
                               contribution_transforms=transformers,
                               feature_descriptions=feature_descriptions)
    lfc.fit()

Finally, we can get the explanation using the ``.produce()`` function. We will also visualize
the most contributing features using the `visualize` model.

.. ipython:: python
    :okwarning:

    # We can now choose an input, and see the model's prediction.
    input_to_explain = x_orig.iloc[0]
    print("Prediction:", lfc.model_predict(input_to_explain)) # Output -> Prediction: [0]

    # We see that this person is not predicted to survive.
    #   Let's see why, by using LocalFeatureContribution's .produce() function
    contributions = lfc.produce(input_to_explain)

    # We can visualize the most contributing features using the real.utils.visualize module.
    #   We will also convert our input to the interpretable space, so we can add it's values to
    #   the visualization
    x_interpret = lfc.convert_data_to_interpretable(input_to_explain)
    visualize.plot_top_contributors(contributions, select_by="absolute", values=x_interpret)

The output will be a bar plot showing the most contributing features, by absolute value.

.. image:: /images/quickstart.png

We can see here that the input passenger's predicted chance of survival was greatly reduced
because of their sex (male) and ticket class (3rd class).
