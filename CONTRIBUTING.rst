.. _contributing:

Contributing to Pyreal
======================

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at the `GitHub Issues page`_.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Pyreal could always use more documentation, whether as part of the
official Pyreal docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at the `GitHub Issues page`_.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up ``pyreal`` for local development.

1. Fork the ``pyreal`` repo on GitHub.
2. Clone your fork locally and cd into the new directory::

    $ git clone git@github.com:your_name_here/pyreal.git
    $ cd pyreal

3. Install the project dependencies using Poetry.
   Set up your fork for local development using the following commands::

    $ poetry install

4. Create a branch for local development. You should always be branching off of
   the ``dev`` branch when contributing::

    $ git checkout -b name-of-your-bugfix-or-feature

   Try to use the naming scheme of prefixing your branch with ``gh-X`` where X is
   the associated issue, such as ``gh-3-fix-foo-bug``. And if you are not
   developing on your own fork, further prefix the branch with your GitHub
   username, like ``githubusername/gh-3-fix-foo-bug``.

   Now you can make your changes locally.

5. While hacking your changes, make sure to cover all your developments with the required
   unit tests, and that none of the old tests fail as a consequence of your changes.
   For this, make sure to run the tests suite and check the code coverage::

    $ poetry run invoke lint       # Check code styling
    $ poetry run invoke test       # Run the tests
    $ poetry run invoke coverage   # Get the coverage report

   All of these commands can be shortened by running within the Poetry shell

    $ poetry shell      # activate the poetry shell
    $ invoke lint       # Check code styling
    $ invoke test       # Run the tests
    $ invoke coverage   # Get the coverage report

6. Make also sure to include the necessary documentation in the code as docstrings following
   the `Google docstrings style`_.
   Test docs are working by running::

    $ poetry run invoke view-docs

7. After running all four commands listed in steps 5 and 6 and
   confirming they work correctly, commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website, merging back into ``dev``.
10. Branches should be deleted on merge.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. It resolves an open GitHub Issue and contains its reference in the title or
   the comment. If there is no associated issue, feel free to create one.
2. Whenever possible, it resolves only **one** issue. If your PR resolves more than
   one issue, try to split it in more than one pull request.
3. The pull request should include unit tests that cover all the changed code.
4. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the documentation in an appropriate place.
5. The pull request should work for all the supported Python versions. Confirm that
   all github actions pass.

Unit Testing Guidelines
-----------------------

All the Unit Tests should comply with the following requirements:

1. Unit Tests should be based only in unittest and pytest modules.

2. The tests that cover a module called ``pyreal/path/to/a_module.py``
   should be implemented in a separated module called
   ``tests/pyreal/path/to/test_a_module.py``.
   Note that the module name has the ``test_`` prefix and is located in a path similar
   to the one of the tested module, just inside the ``tests`` folder.

3. Each method of the tested module should have at least one associated test method, and
   each test method should cover only **one** use case or scenario.

4. Test case methods should start with the ``test_`` prefix and have descriptive names
   that indicate which scenario they cover.
   Names such as ``test_some_methed_input_none``, ``test_some_method_value_error`` or
   ``test_some_method_timeout`` are right, but names like ``test_some_method_1``,
   ``some_method`` or ``test_error`` are not.

5. Each test should validate only what the code of the method being tested does, and not
   cover the behavior of any third party package or tool being used, which is assumed to
   work properly as far as it is being passed the right values.

6. Any third party tool that may have any kind of random behavior, such as some Machine
   Learning models, databases or Web APIs, will be mocked using the ``mock`` library, and
   the only thing that will be tested is that our code passes the right values to them.

7. Unit tests should not use anything from outside the test and the code being tested. This
   includes not reading or writing to any file system or database, which will be properly
   mocked.

Tips
----

To run a subset of tests::

    $ python -m pytest tests.test_global_explanation.py
    $ python -m pytest -k 'foo'

Style guide
-----------------------
Pyreal uses the `PEP 8`_ style guide for all python code:

A few important notes:

1. Indents should be 4 spaces, no tabs

2. Lines should be no more than 99 characters long

3. All functions, classes, and methods should have block comment descriptions using the Google docstring format


Release Workflow
----------------

The process of releasing a new version involves several steps::

1. Create a new branch off of ``dev`` branch called ``release-v.*.*.*``.
2. Update the version in ``setup.cfg``, ``pyreal/__init__.py`` and
   ``HISTORY.md`` files.
3. Make any final small changes needed directly on ``release-v.*.*.*``
4. Make a PR to merge ``release-v.*.*.*`` into ``master``
5. Once merged, tag the merge commit in master, and push the tag.
   This will automatically deploy the release to pypi.
6. Merge ``release-v.*.*.*`` back into ``dev`` with a pull request
7. Make a release on github.com, filling in the release notes with
   a list of pull requests made since the last release.

.. _GitHub issues page: https://github.com/sibyl-dev/pyreal/issues
.. _Google docstrings style: https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments
.. _PEP 8: https://www.python.org/dev/peps/pep-0008/
