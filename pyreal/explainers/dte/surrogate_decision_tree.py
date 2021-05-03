import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression, ElasticNet, LinearRegression

from pyreal.explainers import DecisionTreeExplainerBase
from pyreal.utils.transformer import ExplanationAlgorithm
