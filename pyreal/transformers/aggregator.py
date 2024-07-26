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

    pass
