from enum import Enum, auto


class ExplanationAlgorithm(Enum):
    SHAP = auto()
    SURROGATE_DECISION_TREE = auto()
    PERMUTATION_IMPORTANCE = auto()
