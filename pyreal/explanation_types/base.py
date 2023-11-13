import pandas as pd


def convert_columns_with_dict(df, conversion_dict):
    if (df is None) or (conversion_dict is None):
        return df
    elif isinstance(df, pd.Series):
        return df.rename(conversion_dict)
    else:
        return df.rename(conversion_dict, axis="columns")


class Explanation:
    """
    A type wrapper for outputs from explanation algorithms. Validates that an object is a
    valid explanation output.
    """

    def __init__(self, explanation, values=None):
        """
        Set the wrapped explanation to `explanation` and values to `values` and validate
        Args:
            explanation (object):
                wrapped explanation
            values (DataFrame of shape (n_instances, n_features) or None):
                Values corresponding with the object being explained

        """
        self.explanation = explanation
        self.values = values

        self.validate()
        if self.values is not None:
            if hasattr(values, "ndim") and values.ndim == 1:
                self.values = values.to_frame().T
            self.validate_values()

    def get(self):
        """
        Get the explanation wrapped by this type. Alternative to get_explanation

        Returns:
            object
                wrapped explanation object
        """
        return self.get_explanation()

    def get_explanation(self):
        """
        Get the explanation wrapped by this type

        Returns:
            object
                wrapped explanation object
        """
        return self.explanation

    def get_all(self):
        """
        Get the explanation and wrapped values as a tuple. Convenience function for making a new
        Explanation from a previous one.

        Returns:
            tuple
                Explanation and values
        """
        return self.explanation, self.values

    def get_values(self):
        """
        Return the values associated with the explanation

        Returns:
            DataFrame of shape (n_instances, n_features)
                The values associated with this explanation
        """
        return self.values

    def update_values(self, values, inplace=False):
        """
        Updates this objects values, and validates

        Args:
            values (DataFrame of shape (n_instances, n_features)):
                New values
            inplace (Boolean)
                If True, change the values on this object. Otherwise, create a new object
                identical to this one but with new values

        Returns:
            Explanation
                `self` if `inplace=True`, else the new Explanation object.
        """
        if inplace:
            self.values = values
            self.validate()
            if self.values is not None:
                if hasattr(values, "ndim") and values.ndim == 1:
                    self.values = values.to_frame().T
                self.validate_values()
            return self
        else:
            return self.__class__(self.explanation, values)

    def update_explanation(self, new_explanation, inplace=False, persist_values=True):
        """
        Sets this object's explanation to the new value.

        Args:
            new_explanation (object):
                New explanation
            inplace (Boolean):
                If True, change the explanation on this object. Otherwise, create a new object
                identical to this one but with a new explanation
            persist_values (Boolean):
                If False, remove values from the updated explanation, ie. if the new explanation
                will not be compatible with values
        """
        if inplace:
            self.explanation = new_explanation
            if not persist_values:
                self.values = None
            self.validate()
            if self.values is not None:
                self.validate_values()
            return self
        else:
            if persist_values:
                return self.__class__(new_explanation, self.values)
            else:
                return self.__class__(new_explanation)

    def apply_feature_descriptions(self, feature_descriptions):
        """
        Apply feature descriptions to explanation and values

        Args:
            feature_descriptions (dict):
                Dictionary mapping feature names to interpretable descriptions
        Returns:
            None
        """
        self.update_values(
            convert_columns_with_dict(self.values, feature_descriptions), inplace=True
        )

    def validate(self):
        """
        Validate that `self.explanation` is a valid object of type `Explanation`. If values are not
        None, additionally validate that they are valid values for this Explanation.

        Returns:
            None
        Raises:
            AssertionException
                if `self.explanation` or `self.values` is invalid
        """

    def validate_values(self):
        """
        Validate that `self.values` are valid values for this Explanation.

        Returns:
            None
        Raises:
            AssertionException
                if `self.values` is invalid
        """
        if not isinstance(self.values, pd.DataFrame) and not isinstance(self.values, pd.Series):
            raise AssertionError("values must be of type DataFrame")
