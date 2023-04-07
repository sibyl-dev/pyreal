# 📊 Generating Explanations

Explanations can be generated using the `RealApp` class's `.produce()` functions using the name of the explanation type you desire.

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

Similarly, you can generate other kinds of explanations using the functions found in our [Glossary of Explanations](../../glossary-of-explantions.md).

&#x20;
