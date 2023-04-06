# 📊 Generating Explanations

Explanations can be generated using the `RealApp` class's `.produce()` functions using the name of the explanation type you desire.

For **local explanations**, or explanations of a specific model prediction, `.produce()` takes in a required argument of the data row(s) you would like the explain. For **global explanations**, or explanations of the model in general, no argument is required.

<pre class="language-python"><code class="lang-python">from pyreal.sample_applications import student_performance

real_app = student_performance.load_app()
students = student_performance.load_students()

<strong>all_explanations = realApp.produce_feature_contributions(students)
</strong></code></pre>

all\_explanations can be indexed by row ids, found in the column labelled by the optional `id_column` parameter to RealApps (if no id column is provided, the ids are row number).&#x20;

This allows us to access the explanation for a given student by name:

```python
all_explanations["Trevor Butler"]
```

... which outputs a DataFrame with all features, their contributions, and their average or mode value:

| Feature Name                   | Feature Value | Contribution | Average/Mode |
| ------------------------------ | ------------- | ------------ | ------------ |
| Age                            | 17            | -0.13377     | 16.54545     |
| Mother's education             | 3             | 0.001509     | 3.090909     |
| Father's education             | 2             | -0.05293     | 2.181818     |
| Weekly study time              | 2             | 0.023261     | 2            |
| Number of past class failures  | 2             | -3.74398     | 0.272727     |
| Extra education support        | yes           | -0.59027     | no           |
| Wants to take higher education | yes           | 0.065789     | yes          |
| ...                            | ...           | ...          | ...          |

&#x20;
