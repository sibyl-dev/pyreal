# Making Contributions

### Contribution Process

Ready to contribute? Here's how to setup Pyreal for local development, and make contributions

* Fork the `pyreal` repo on GitHub.
* Clone your fork locally and cd into the new directory:

```git
git clone git@github.com:your_name_here/pyreal.git
cd pyreal
git checkout dev
```

* Install the project dependencies using Poetry.

```
poetry install -E examples
```

* Create a branch for local development. You should always be branching off of the `dev` branch when contributing.&#x20;

```git
git checkout -b name-of-your-bugfix-or-feature
```

* Now you can make your changes locally. While making your changes, make sure to cover all your additions with the [required unit tests](unit-testing-guidelines.md), and that none of the old tests fail as a consequence of your changes. For this, make sure to run the tests suite and check the code coverage:

```
poetry run invoke lint       # Check code styling
poetry run invoke test       # Run the tests
poetry run invoke coverage   # Get the coverage report
```

All of these commands can be shortened by running within the Poetry shell::

```
poetry shell      # activate the poetry shell
invoke lint       # Check code styling
invoke test       # Run the tests
invoke coverage   # Get the coverage report
```

* Make also sure to include the necessary documentation in the code as docstrings following the [Google docstrings style guide.](https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments) Test the autodocs generation process by running:

```
poetry run invoke view-docs
```

* Any PR that includes a new `Explainer` class, significant changes to an existing `Explainer` class, or significant changes to the explanation workflow as a whole should also include the results of a benchmarking run. See [Benchmarking Guidelines](benchmarking-guidelines.md) for more info.
* After running all test and docs commands listed above and confirming they work correctly, commit your changes and push your branch to GitHub:

```
git add .
git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature
```

* Submit a pull request through the GitHub website, merging back into `dev`.
* Once you have a minimum of two approvals, you can merge your branch in. Branches should be deleted on merge.