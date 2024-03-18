---
description: Understanding your ML model and its predictions.
---

# Generating Explanations

Sometimes you may want more information about how an ML model came up with its prediction on an input. Or, you may have questions about the ML model in general. For our example application of predicting house prices based on information about houses, you may have questions like:

1. What information about the house (or _features_) contributed to the prediction of $102,300?
2. Have we seen houses in the past similar to this one, and what were their prices?
3. What features, in general, does the model consider most important for predicting house prices?
4. How does the model use the "size" feature? Are bigger houses always predicted to be more expensive?

You can answer these questions by look at ML model explanations. Explanations are generated using the `RealApp` class's `.produce()` functions using the name of the explanation type you desire.

{% hint style="info" %}
If you are not familiar with the concept of ML explanations, you can see the [Introduction to Explanations](../../further-reading/introduction-to-explanations.md) guide for more details.
{% endhint %}

For **local explanations**, or explanations of a specific model prediction, `.produce()` takes in a required argument of the data row(s) you would like the explain. For **global explanations**, or explanations of the model in general, no argument is required.

<pre class="language-python"><code class="lang-python">from pyreal.sample_applications import student_performance

real_app = student_performance.load_app()
students = student_performance.load_students()

<strong>all_explanations = realApp.produce_feature_contributions(students)
</strong></code></pre>

`all_explanations` can be indexed by row ids, found in the column labelled by the optional `id_column` parameter to RealApps (if no id column is provided, the ids are row number).&#x20;

This allows us to access the explanation for a given student by name:

```python
all_explanations["Trevor Butler"]
```

... which outputs a DataFrame with all features, their contributions, and their average or mode value:

| Feature Name                   | Feature Value | Contribution | Average/Mode |
| ------------------------------ | ------------- | ------------ | ------------ |
| Age                            | 17            | -0.13        | 16.54        |
| Mother's education             | 3             | 0.0015       | 3.09         |
| Father's education             | 2             | -0.052       | 2.18         |
| Weekly study time              | 2             | 0.023        | 2            |
| Number of past class failures  | 2             | -3.74        | 0.27         |
| Extra education support        | yes           | -0.59        | no           |
| Wants to take higher education | yes           | 0.065        | yes          |
| ...                            | ...           | ...          | ...          |

Again, for global explanations you do not need a parameter for `.produce()`:

```python
from pyreal.sample_applications import student_performance

real_app = student_performance.load_app()
students = student_performance.load_students()

explanation = realApp.produce_feature_importance()
```

... which generates a DataFrame of feature importance values

| Feature Name                   | Importance |
| ------------------------------ | ---------- |
| Age                            | 3.4        |
| Mother's education             | .002       |
| Father's education             | .023       |
| Weekly study time              | .54        |
| Number of past class failures  | 9.7        |
| Extra education support        | 1.45       |
| Wants to take higher education | .23        |
| ...                            | ...        |

Similarly, you can generate other kinds of explanations using the functions found in our [Glossary of Explanations](../../glossaries/glossary-of-explantions.md).
