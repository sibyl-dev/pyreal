---
description: Understanding the language of Pyreal
---

# Terminology guide

Here, we offer a short summary of the specific terminology Pyreal used throughout the Pyreal development guides and codebase.

### General Concepts

<table><thead><tr><th width="225">Term</th><th>Definition</th></tr></thead><tbody><tr><td><strong>model</strong></td><td>A machine learning predictor, defined as an object with a <code>.predict()</code> function.</td></tr><tr><td><strong>x</strong></td><td> Data, in the form of a pandas DataFrame, usually of shape (<code>n_instances</code>, <code>n_features</code>).</td></tr><tr><td><strong>x_train</strong></td><td>Training data for the model and explanation algorithms</td></tr><tr><td><strong>explanation</strong></td><td>An explanation of a model or model prediction</td></tr><tr><td><strong>local explanation</strong></td><td>An explanation of a specific model prediction</td></tr><tr><td><strong>global explanation</strong></td><td>An explanation of a model overall</td></tr><tr><td><strong>feature space</strong></td><td>A specific format of data (for example, ML models take in data in a specific feature space)</td></tr></tbody></table>

### Pyreal Classes

<table><thead><tr><th width="224">Term</th><th>Definition</th></tr></thead><tbody><tr><td><strong>RealApp</strong></td><td>Pyreal objects that hold important information and generate various types of interpretable explanations</td></tr><tr><td><strong>Explainer</strong></td><td>Pyreal objects that take in data and a model and return a specific type of explanation</td></tr><tr><td><strong>Transformer</strong></td><td>Pyreal objects that transform data and explanations from one feature space to another</td></tr><tr><td><strong>Explanation</strong></td><td>Pyreal objects that hold information about a specific generated explanation, as produced by Explainers</td></tr></tbody></table>

### Feature Spaces

<table><thead><tr><th width="226">Term</th><th>Definition</th></tr></thead><tbody><tr><td><strong>x_orig, explanation_orig</strong></td><td>Data/explanation using the original feature space (whatever feature space the data starts in).</td></tr><tr><td><strong>x_algorithm, explanation_algorithm</strong></td><td>Data/explanation using the feature space expected by the explanation algorithm (explanation-ready feature space)</td></tr><tr><td><strong>x_model, explanation_model</strong></td><td>Data/explanation using the feature space the model expects</td></tr><tr><td><strong>x_interpret, explanation_interpret</strong></td><td>Data/explanation using the most human-readable feature space available</td></tr><tr><td><strong>model transformers</strong></td><td>Transformers that transform data between the original feature space to the model-ready feature space. Transformers with <code>model=True</code> flag.</td></tr><tr><td><strong>algorithm transformers</strong></td><td>Transformers that transform data and explanations between the original feature space to the explanation-algorithm-ready feature spaces. Transformers with <code>algorithm=True</code> flag.</td></tr><tr><td><strong>interpret transformers</strong></td><td>Transformers that transform data and explanations between the original feature space to the interpretable feature space. Transformers with <code>interpret=True</code> flag.</td></tr></tbody></table>
