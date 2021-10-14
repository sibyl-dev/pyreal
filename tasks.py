import subprocess
import webbrowser
import shutil

from pathlib import Path
from invoke import task
from sys import executable
import os


def _rm_recursive(path: Path, pattern: str):
    """
    Glob the given relative pattern in the directory represented by this path,
        calling shutil.rmtree on them all
    """

    for p in path.glob(pattern):
        shutil.rmtree(p, ignore_errors=True)


@task
def clean_build(context):
    """
    Cleans the build
    """

    shutil.rmtree(Path("build"), ignore_errors=True)
    shutil.rmtree(Path("dist"), ignore_errors=True)
    shutil.rmtree(Path(".eggs"), ignore_errors=True)

    _rm_recursive(Path("."), "**/*.egg-info")
    _rm_recursive(Path("."), "**/*.egg")


@task
def clean_coverage(context):
    """
    Cleans the coverage results
    """

    Path(".coverage").unlink(missing_ok=True)

    for path in Path(".").glob(".coverage.*"):
        path.unlink(missing_ok=True)

    shutil.rmtree(Path("htmlcov"), ignore_errors=True)


@task
def clean_docs(context):

    for path in Path("docs/api").glob("*.rst"):
        path.unlink(missing_ok=True)

    subprocess.run(["sphinx-build", "-M", "clean", ".", "_build"], cwd=Path("docs"))


@task
def clean_pyc(context):
    """
    Cleans compiled files
    """

    _rm_recursive(Path("."), "**/*.pyc")
    _rm_recursive(Path("."), "**/*.pyo")
    _rm_recursive(Path("."), "**/*~")
    _rm_recursive(Path("."), "**/__pycache__")


@task
def clean_test(context):
    """
    Cleans the test store
    """

    shutil.rmtree(Path(".pytest_cache"), ignore_errors=True)


@task
def coverage(context):
    """
    Runs the unit test coverage analysis
    """

    subprocess.run(["coverage", "run", "--source", "pyreal", "-m", "pytest"])
    subprocess.run(["coverage", "report", "-m"])
    subprocess.run(["coverage", "html"])

    url = os.path.join("htmlcov", "index.html")
    webbrowser.open(url)


@task
def docs(context):
    """
    Cleans the doc builds and builds the docs
    """

    clean_docs(context)

    subprocess.run(["sphinx-build", "-b", "html", ".", "_build"], cwd=Path("docs"))


@task
def fix_lint(context):
    """
    Fixes all linting and import sort errors. Skips init.py files for import sorts
    """

    subprocess.run(["autoflake", "--in-place", "--recursive",
                   "--remove-all-unused-imports", "--remove-unused-variables", "pyreal"])
    subprocess.run(["autoflake", "--in-place", "--recursive",
                   "--remove-all-unused-imports", "--remove-unused-variables", "tests"])
    subprocess.run(["autopep8", "--in-place", "--recursive", "--aggressive", "pyreal", "tests"])
    subprocess.run(["isort", "--atomic", "pyreal", "tests", "--skip", "__init__.py"])


@task
def lint(context):
    """
    Runs the linting and import sort process on all library files and tests and prints errors.
        Skips init.py files for import sorts
    """
    subprocess.run(["flake8", "pyreal", "tests"])
    subprocess.run(["isort", "-c", "pyreal", "tests", "--skip", "__init__.py"])


@task
def test(context):
    """
    Runs all test commands.
    """

    test_unit(context)

    test_readme(context)

    test_tutorials(context)


@task
def test_readme(context):
    """
    Runs all scripts in the README and checks for exceptions
    """

    # Resolve the executable with what's running the program. It could be a venv
    subprocess.run([executable, "-m", "doctest", "-v", "README.md"])


@task
def test_tutorials(context):
    """
    Runs all scripts in the tutorials directory and checks for exceptions
    """

    subprocess.run(["pytest", "--nbmake", "./tutorials", "-n=auto"])


@ task
def test_unit(context):
    """
    Runs all unit tests and outputs results and coverage
    """
    subprocess.run(["pytest", "--cov=pyreal"])


@ task
def view_docs(context):
    """
    Opens the docs in a browser window
    """

    docs(context)

    url = os.path.join("docs", "_build", "index.html")
    webbrowser.open(url)
