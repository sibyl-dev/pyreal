---
description: Understanding the language of Pyreal
---

# Developer Terminology Guide

Here, we offer a short summary of the specific terminology Pyreal used throughout the Pyreal development guides and codebase.

### General Concepts

| Term              | Definition                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------- |
| **model**         | A machine learning predictor, defined as an object with a `.predict()` function.            |
| **x**             |  Data, in the form of a pandas DataFrame, usually of shape (`n_instances`, `n_features`).   |
| **x\_train**      | Training data for the model and explanation algorithms                                      |
| **explanation**   | An explanation of a model or model prediction                                               |
| **feature space** | A specific format of data (for example, ML models take in data in a specific feature space) |

### Pyreal Classes

| Term            | Definition                                                                                              |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| **RealApp**     | Pyreal objects that hold important information and generate various types of interpretable explanations |
| **Explainer**   | Pyreal objects that take in data and a model and return a specific type of explanation                  |
| **Transformer** | Pyreal objects that transform data and explanations from one feature space to another                   |
| **Explanation** | Pyreal objects that hold information about a specific generated explanation, as produced by Explainers  |

### Feature Spaces

| Term                                     | Definition                                                                                                                                                                       |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **x\_orig, explanation\_orig**           | Data/explanation using the original feature space (whatever feature space the data starts in).                                                                                   |
| **x\_algorithm, explanation\_algorithm** | Data/explanation using the feature space expected by the explanation algorithm (explanation-ready feature space)                                                                 |
| **x\_model, explanation\_model**         | Data/explanation using the feature space the model expects                                                                                                                       |
| **x\_interpret, explanation\_interpret** | Data/explanation using the most human-readable feature space available                                                                                                           |
| **model transformers**                   | Transformers that transform data between the original feature space to the model-ready feature space. Transformers with `model=True` flag.                                       |
| **algorithm transformers**               | Transformers that transform data and explanations between the original feature space to the explanation-algorithm-ready feature spaces. Transformers with `algorithm=True` flag. |
| **interpret transformers**               | Transformers that transform data and explanations between the original feature space to the interpretable feature space. Transformers with `interpret=True` flag.                |
