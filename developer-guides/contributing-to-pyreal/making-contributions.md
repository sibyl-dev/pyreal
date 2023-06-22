# Making Contributions

### Contribution Process

Ready to contribute? Here's how to setup Pyreal for local development, and make contributions

* Clone the `pyreal` repo onto your local machine and cd into the new directory:

```git
git clone https://github.com/sibyl-dev/pyreal.git
cd pyreal
git checkout dev
```

* Install the project dependencies using Poetry.

```
poetry install
```

* Create a branch for local development. You should always be branching off of the `dev` branch when contributing. Give your branch a descriptive name that describes what change you will be making. Each branch should aim to resolve at most one issue.

```git
git checkout -b name-of-your-bugfix-or-feature
```

* Now you can make your changes locally. While making your changes, make sure to cover all your additions with the [required unit tests](unit-testing-guidelines.md), and that none of the old tests fail as a consequence of your changes. For this, make sure to run the tests suite and check the code coverage:

```bash
poetry run invoke lint       # Check code styling
poetry run invoke test       # Run the tests
poetry run invoke coverage   # Get the coverage report

# Other helpful commands:
poetry run invoke fix-lint   # Run the auto-fomatter (fixes most linting errors)
poetry run invoke test-unit  # Run only unit tests
poetry run invoke test-tutorials # Run only the tutorial scripts
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
* Once you have a minimum of two approvals, you can merge your branch in. **Branches should be deleted on merge.**
