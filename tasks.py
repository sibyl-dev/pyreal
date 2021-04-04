import glob
import os
from pathlib import Path
import shutil

from invoke import task


@task
def tutorials(c):
    for ipynb_file in glob.glob("tutorials/*.ipynb") + glob.glob("tutorials/**/*.ipynb"):
        if ".ipynb_checkpoints" not in ipynb_file:
            c.run(
                (
                    "jupyter nbconvert --execute --ExecutePreprocessor.timeout=3600 "
                    "--to=html --stdout {ipynb_file}".format(ipynb_file=ipynb_file)
                ),
                hide="out",
            )


@task
def readme(c):
    test_path = Path("tests/readme_test")
    if test_path.exists() and test_path.is_dir():
        shutil.rmtree(test_path)

    cwd = os.getcwd()
    os.makedirs(test_path, exist_ok=True)
    shutil.copy("README.md", test_path / "README.md")
    os.chdir(test_path)
    c.run("rundoc run --single-session python3 -t python3 README.md")
    os.chdir(cwd)
    shutil.rmtree(test_path)
