import pandas as pd

from pyreal.transformers.base import TransformerBase


def _generate_many_to_one(one_to_many):
    many_to_one = {}
    for parent in one_to_many:
        for child in one_to_many[parent]:
            many_to_one[child] = parent

    return many_to_one


def _generate_one_to_many(many_to_one):
    one_to_many = {}
    for child in many_to_one:
        parent = many_to_one[child]
        if parent not in one_to_many:
            one_to_many[parent] = [child]
        else:
            one_to_many[parent].append(child)
    return one_to_many


def _generate_from_df(df):
    one_to_many = df.groupby("parent")["child"].apply(list).to_dict()
    return one_to_many


class Mappings:
    def __init__(self, one_to_many, many_to_one):
        """
        Initialize a new mappings object
        For common use, use Mappings.generate_mapping()

        Args:
            one_to_many (dictionary):
                {parent_feature_name : [child_feature_name_1], ...}
            many_to_one (dictionary):
                {child_feature_name_1 : parent_feature_name, ...}
        """

        self.one_to_many = one_to_many
        self.many_to_one = many_to_one

    @staticmethod
    def generate_mappings(one_to_many=None, many_to_one=None, dataframe=None):
        """
        Generate a new Mappings object using one of the input formats
        All but one keyword should be None

        Args:
            one_to_many:
                {parent_feature_name : [child_feature_name_1], ...}
            many_to_one:
                {child_feature_name_1 : parent_feature_name, ...}
            dataframe:
                DataFrame with three columns named [parent, child]
                ie., [["Feat1", "feat1a"],
                      ["Feat1", "feat1b"]]
        Returns:
            Mappings
                A Mappings objects representing the column relationships
        """

        if one_to_many is not None:
            return Mappings(one_to_many, _generate_many_to_one(one_to_many))
        if many_to_one is not None:
            return Mappings(_generate_one_to_many(many_to_one), many_to_one)
        if dataframe is not None:
            one_to_many = _generate_from_df(dataframe)
            return Mappings(one_to_many, _generate_many_to_one(one_to_many))


class Aggregator(TransformerBase):
    """
    Aggregate features into a single parent feature class
    """

    def __init__(self, mappings, func="sum", drop_original=True, missing="ignore", **kwargs):
        """
        Initialize a new Aggregator object

        Args:
            mappings (Mappings):
                A Mappings object representing the column relationships
            func (callable or one of ["sum", "mean", "max", "min", "remove"]):
                The function to use to aggregate the features. If set to "remove", the parent
                feature will be given None values (use in cases where you want to aggregate
                explanations on features, but no valid aggregation exists)
            drop_original (bool):
                Whether to drop the original features after aggregation
            missing (str):
                How to handle values in the mappings but not the transform data.
                One of ["ignore", "raise"]
                If "ignore", parent features will be made out of any child features that exist. If
                no child features exist, the parent feature will not be added.
                If "raise", an error will be raised if any child features are missing
        """

        def mean(x):
            return sum(x) / len(x)

        def remove(x):
            return None

        self.mappings = mappings
        if func == "sum":
            func = sum
        elif func == "mean":
            func = mean
        elif func == "max":
            func = max
        elif func == "min":
            func = min
        elif func == "remove":
            func = remove
        self.func = func
        self.drop_original = drop_original
        self.missing = missing
        super().__init__(**kwargs)

    def data_transform(self, X):
        """
        Transform the input data, aggregating the features as specified in the mappings

        Args:
            X (DataFrame):
                The input data

        Returns:
            DataFrame of shape (n_samples, n_features_new):
                The transformed data with the aggregated features
        """
        X = X.copy()
        for parent in self.mappings.one_to_many:
            children = self.mappings.one_to_many[parent]
            if self.missing == "raise":
                if not all(child in X for child in self.mappings.one_to_many[parent]):
                    raise ValueError("Missing child features")
            if self.missing == "ignore":
                if not any(child in X for child in self.mappings.one_to_many[parent]):
                    continue
                else:
                    children = [child for child in self.mappings.one_to_many[parent] if child in X]
            X[parent] = X[children].apply(self.func, axis=1)
        if self.drop_original:
            X = X.drop(columns=list(self.mappings.many_to_one.keys()), errors="ignore")
        return X

    def transform_explanation_additive_feature_contribution(self, explanation):
        """
        Sum together the contributions in explanation from the child features to get the
        parent features

        Args:
            explanation (AdditiveFeatureContributionExplanation):
                The explanation to transform
        """
        return explanation.update_explanation(
            _helper_sum_columns(explanation.get(), self.mappings, self.missing)
        )

    def transform_explanation_additive_feature_importance(self, explanation):
        """
        Sum together the importances in explanation from the child features to get the
        parent features

        Args:
            explanation (AdditiveFeatureImportanceExplanation):
                The explanation to transform
        """
        return explanation.update_explanation(
            _helper_sum_columns(explanation.get(), self.mappings, self.missing)
        )


def _helper_sum_columns(df, mappings, missing):
    # Remove any children not present in df
    if missing == "ignore":
        one_to_many = {
            parent: [child for child in children if child in df]
            for parent, children in mappings.one_to_many.items()
        }

        one_to_many = {parent: children for parent, children in one_to_many.items() if children}
    else:
        one_to_many = mappings.one_to_many

    if missing == "raise":
        if not all(child in df for children in one_to_many.values() for child in children):
            raise ValueError("Missing child features")

    mappings = Mappings.generate_mappings(one_to_many=one_to_many)

    parents = pd.DataFrame(
        {parent: df[children].sum(axis=1) for parent, children in mappings.one_to_many.items()}
    )
    return pd.concat([df.drop(columns=list(mappings.many_to_one.keys())), parents], axis=1)
