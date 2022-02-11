.. _transformer_workflow:
..    include:: <isonum.txt>

Transformer Workflow
=====================

Terminology
-----------
Pyreal introduces specific terms and naming schemes to refer to different feature spaces and
transformations. Here, we offer a short summary of these terms. You can see examples of
some of these in the quick start tutorial above.

+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| Term                                    | Description                                                                                                                                             |
+=========================================+=========================================================================================================================================================+
| ``model``                               | A machine learning predictor, defined as an object with a ``.predict()`` funtion                                                                        |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``x``, ``x_train``                      | Data, in the form of a pandas DataFrame                                                                                                                 |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``explanation``                         | An explanation of a model or model prediction                                                                                                           |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Explainer``                           | Pyreal objects that take in data and a model and return an explanation                                                                                  |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Transformer``                         | Pyreal objects that transform data and explanations from one feature space to another                                                                   |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``x_orig``, ``explanation_orig``        | Data or an explanation using the original feature space (whatever feature space the data starts in)                                                     |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``x_algorithm``, ``explanation_algorithm``  | Data or an explanation using the feature space expected by the explanation algorithm (explanation-ready feature space)                                  |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``x_model``, ``explanation_model``      | Data or an explanation using the feature space the model expects                                                                                        |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``x_interpret``,                        |                                                                                                                                                         |
| ``explanation_interpret``               | Data or an explanation using the most human-readable feature space available                                                                            |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``model transformers``                  | Transformers that transform data between the original feature space to the model-ready feature space. Transformers with ``model=True`` flag.            |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``algorithm transformers``              | Transformers that transform data and explanations between the original feature space to the explanation-algorithm-ready feature spaces. Transformers with ``algorithm=True`` flag.    |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``i_transformers``                      | Transformers that transform data and explanations between the original feature space to the interpretable feature space. Transformers with ``interpret=True`` flag. |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+

Workflow
----------
To generate interpretable explanations, Pyreal follows the following series of transforms:

- ``x_orig`` |rarr| **algorithm_transformers.transform()** |rarr| ``x_algorithm``
- ``x_algorithm`` |rarr| **Explainer.produce()** |rarr| ``explanation_algorithm``

  - ``x_algorithm`` |rarr| **model_transformers.transform()** |rarr| ``x_model``

- ``explanation_algorithm`` |rarr| **algorithm_transformers.inverse_transform_explanation()** |rarr| ``explanation_orig``
- ``explanation_orig`` |rarr| **interpret_transformers.transform_explanation()** |rarr| ``explanation_interpret``
- ``x_orig`` |rarr| **interpret_transformers.transform()** |rarr| ``x_interpret``



