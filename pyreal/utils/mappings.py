def generate_one_hot_to_categorical(categorical_to_one_hot):
    one_hot_to_categorical = {}
    for cf in categorical_to_one_hot:
        for ohf in categorical_to_one_hot[cf]:
            one_hot_to_categorical[ohf] = (cf, categorical_to_one_hot[cf][ohf])

    return one_hot_to_categorical


def generate_categorical_to_one_hot(one_hot_to_categorical):
    categorical_to_one_hot = {}
    for ohf in one_hot_to_categorical:
        cf = one_hot_to_categorical[ohf][0]
        value = one_hot_to_categorical[ohf][1]
        if cf not in categorical_to_one_hot:
            categorical_to_one_hot[cf] = {ohf: value}
        else:
            categorical_to_one_hot[cf][ohf] = value
    return categorical_to_one_hot


def generate_from_df(df):
    # TODO: rename columns to be more natural
    categorical_to_one_hot = {}
    for i in range(df.shape[0]):
        cf = df["name"][i]
        ohf = df["original_name"][i]
        value = df["value"][i]
        if cf not in categorical_to_one_hot:
            categorical_to_one_hot[cf] = {ohf: value}
        else:
            categorical_to_one_hot[cf][ohf] = value
    return categorical_to_one_hot


class Mappings:
    def __init__(self, categorical_to_one_hot, one_hot_to_categorical):
        """
        Initialize a new mappings object
        For common use, use Mappings.generate_mapping()

        :param categorical_to_one_hot: dictionary
               {categorical_feature_name : {OHE_feature_name : value, ...}, ... }
        :param one_hot_to_categorical: dictionary
               {OHE_feature_name : (categorical_feature_name, value), ...}
        """
        self.categorical_to_one_hot = categorical_to_one_hot
        self.one_hot_to_categorical = one_hot_to_categorical

    @staticmethod
    def generate_mappings(categorical_to_one_hot=None,
                          one_hot_to_categorical=None,
                          dataframe=None):
        """
        Generate a new Mappings object using one of the input formats
        One one keyword should be None

        :param categorical_to_one_hot: dictionary
               {categorical_feature_name : {OHE_feature_name : value, ...}, ... }
        :param one_hot_to_categorical:
               {OHE_feature_name : (categorical_feature_name, value), ...}
        :param dataframe:
               DataFrame # TODO: specify type
        :return:
        """
        if categorical_to_one_hot is not None:
            return Mappings(categorical_to_one_hot,
                            generate_one_hot_to_categorical(categorical_to_one_hot))
        if one_hot_to_categorical is not None:
            return Mappings(generate_categorical_to_one_hot(one_hot_to_categorical),
                            one_hot_to_categorical)
        if dataframe is not None:
            categorical_to_one_hot = generate_from_df(dataframe)
            return Mappings(categorical_to_one_hot,
                            generate_one_hot_to_categorical(categorical_to_one_hot))
