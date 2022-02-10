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
| ``x_explain``, ``explanation_explain``  | Data or an explanation using the feature space expected by the explanation algorithm (explanation-ready feature space)                                  |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``x_model``, ``explanation_model``      | Data or an explanation using the feature space the model expects                                                                                        |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``x_interpret``,                        |                                                                                                                                                         |
| ``explanation_interpret``               | Data or an explanation using the most human-readable feature space available                                                                            |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``e_transformers``                      | Transformers that transform data from the original to the explanation-ready, and explanations from the explanation-ready to the original feature space  |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``m_transformers``                      | Transformers that transform data between the explanation-ready and model-ready feature spaces                                                           |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``i_transformers``                      | Transformers that transform data and explanations from the original feature space to the interpretable feature space                                    |
+-----------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+

Workflow
----------
To generate interpretable explanations, Pyreal follows the following series of transforms:

- ``x_orig`` |rarr| **e\_transformers.transform()** |rarr| ``x_explain``
- ``x_explain`` |rarr| **Explainer.produce()** |rarr| ``explanation_explain``

  - ``x_explain`` |rarr| **m\_transformers.transform()** |rarr| ``x_model``

- ``explanation_explain`` |rarr| **e\_transformers.inverse_transform_explanation()** |rarr| ``explanation_orig``
- ``explanation_orig`` |rarr| **i\_transformers.transform_explanation()** |rarr| ``explanation_interpret``
- ``x_orig`` |rarr| **i\_transformers.transform()** |rarr| ``x_interpret``



