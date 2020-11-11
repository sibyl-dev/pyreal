#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup


with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    # Math
    'numpy>=1.8',
    'pandas>=1.0.3',
    "scikit-learn>=0.22",
    "shap>=0.36.0",
    "eli5>=0.10",
    'matplotlib>=3.2.1'
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    'jupyter>=1.0.0,<2'
]

development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r2>=0.2.5,<0.3',
    'nbsphinx>=0.5.0,<0.7',
    'Sphinx>=3,<4',
    'pydata-sphinx-theme',
    'autodocsumm>=0.1.10',
    'PyYaml>=5.3.1,<6',
    'argh>=0.26.2,<1',
    'ipython>7.18.0',

    # style check
    'flake8>=3.7.7',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.2',
    'autopep8>=1.4.3',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1',
    'tox>=2.9.1',
]

examples_require = [
    "keras>=2.4.3",
    "tensorflow>=2.2",
]

setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description='Library for evaluating and deploying machine learning explanations.',
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
        'examples': examples_require,
    },
    install_package_data=True,
    install_requires=install_requires,
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='pyreal Pyreal',
    name='pyreal',
    packages=find_packages(include=['pyreal', 'pyreal.*']),
    python_requires='>=3.4',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/DAI-Lab/pyreal',
    version='0.1.0',
    zip_safe=False,
)
