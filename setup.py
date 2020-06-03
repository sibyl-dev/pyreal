#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    # Common libs
    'termcolor==1.1.0',
    'PyYAML==5.1',

    # Math
    'numpy>=1.8',
    'pandas>=1.0.3',
    "scikit-learn>=0.22",
    "shap>=0.35",
    "eli5>=0.10",

    # Flask
    'Flask==1.0.2',
    'Flask-Cors==3.0.7',
    'Flask-RESTful==0.3.7',
    'Werkzeug==0.15.3',
    'gevent==1.2.2',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    
    # ------------- Flask --------------- #
    'pytest-flask>=0.14.0',
    'pytest-xdist>=1.25.0'
]

development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r>=0.2.0',
    'Sphinx>=1.7.1',
    'sphinx_rtd_theme>=0.2.4',
    'autodocsumm>=0.1.10',

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
    },
    entry_points={
        'console_scripts': [
            'met=explanation_toolkit.cli:main',
        ],
    },
    install_package_data=True,
    install_requires=install_requires,
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='explanation_toolkit explanation-toolkit Explanation Toolkit',
    name='explanation-toolkit',
    packages=find_packages(include=['explanation_toolkit', 'explanation_toolkit.*']),
    python_requires='>=3.4',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/DAI-Lab/explanation-toolkit',
    version='0.1.0.dev0',
    zip_safe=False,
)
