import subprocess
import webbrowser
import shutil

from pathlib import Path
from invoke import task


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

    subprocess.run(["make", "clean"], cwd=Path("docs"), shell=True)


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

    url = Path("htmlcov/index.html").absolute()
    webbrowser.open(url)


@task
def docs(context):
    """
    Cleans the doc builds and builds the docs
    """

    clean_docs(context)

    subprocess.run(["make", "html"], cwd=Path("docs"), shell=True)


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

    test_path = Path('tests/readme_test')
    shutil.rmtree(test_path, ignore_errors=True)

    test_path.mkdir(parents=True, exist_ok=True)
    shutil.copy('README.md', test_path / 'README.md')

    subprocess.run(["rundoc", "run", "--single-session", "python3",
                   "-t", "python3", "README.md"], cwd=test_path)
    shutil.rmtree(test_path)


@task
def test_tutorials(context):
    """
    Runs all scripts in the tutorials directory and checks for exceptions
    """

    for ipynb_file in Path("tutorials").glob('**/*.ipynb'):
        checkpoints = ipynb_file.parents[0] / '.ipynb_checkpoints'
        if not checkpoints.is_file():
            subprocess.run(["jupyter", "nbconvert", "--execute",
                            "--ExecutePreprocessor.timeout=60",
                            "--to=html", "--stdout", f"{ipynb_file}"], stdout=subprocess.DEVNULL)


@task
def test_unit(context):
    """
    Runs all unit tests and outputs results and coverage
    """
    subprocess.run(["pytest", "--cov=pyreal"])


@task
def view_docs(context):
    """
    Opens the docs in a browser window
    """

    docs(context)

    url = Path("docs/_build/html/index.html").absolute()
    webbrowser.open(url)
