# Benchmarking Guidelines

### **When to run benchmarking:**

PR's should include results and logs from a new benchmarking run whenever:

1. A PR includes a new `Explainer` class. In this case, a new benchmark `Challenge` class must also be added (see below).
2. A PR includes _significant_ changes to an existing `Explainer` class (when in doubt, ask reviewers).
3. A PR includes _significant_ changes to the general fit-produce explanation workflow.

It's a good idea to run the benchmarking procedure for all PRs, as it can catch subtle bugs that may be missed by other tests (if this happens, it should be reported in a Github issue, so more tests can be added). However, unless a PR falls under one of the categories listed above, results and logs should **not** be pushed to the repo.

### **How to run benchmarking**

The benchmarking process can be run using:

```
$ poetry run invoke benchmark
```

This will run the process, and save the results to `pyreal/benchmark/results`.

To run the benchmarking process without leaving a results directory (ie, for testing):

```
$ poetry run invoke benchmark-no-log
```

This will run the process, and delete the results directory at the end.

To run the benchmarking process while downloading the benchmark datasets locally (this will speed up future runs):

```
$ poetry run invoke benchmark-download
```

### **Adding challenges**

If your PR adds a new `Explainer` class, you must add a corresponding `Challenge` class in the same PR. To do so, follow these steps:

1. Add a file called `[$explainer_name]_challenge.py` to the corresponding place in `pyreal/benchmark/challenges`.
2. Fill this file out using this template, following the example of the others::

```python
$ class [$ExplainerName]Challenge(ExplainerChallenge):
$    def create_explainer(self):
$        return [$ExplainerName](model=self.dataset.model, x_train_orig=self.dataset.X,
                                 transformers=self.dataset.transforms, fit_on_init=True)
```

&#x20;3\. Add the new challenge to `pyreal/benchmark/main.get_challenges()`
