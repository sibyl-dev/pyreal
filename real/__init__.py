# -*- coding: utf-8 -*-

"""Top-level package for Explanation Toolkit."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.0.dev0'

from real.explainers import Explainer, LocalFeatureContribution, ShapFeatureContribution
from real.explainers.local_feature_explanation import \
    fit_local_feature_contributions, fit_and_produce_local_feature_contributions, \
    produce_feature_contributions
