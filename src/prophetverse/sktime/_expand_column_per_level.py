import re

import numpy as np
import pandas as pd
from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose import TransformerPipeline


class ExpandColumnPerLevel(BaseTransformer):
    """
    Transformer that expands columns based on a regular expression pattern per level of a multi-level index.
    If there are M colums matching the pattern and N distinct multi-level indexes, then the transformer will
    create M * N new columns, setting the value to zero to each of M * (N-1) columns for a given index j if the current index
    i /= j.

    Parameters:
    - columns_regex (list): A list of regular expression patterns to match the columns to be expanded.

    Attributes:
    - matched_columns_ (set): A set of columns that match the regular expression patterns.
    - new_columns_ (dict): A dictionary mapping the new column names to the original column names.

    Methods:
    - fit(X, y=None): Fit the transformer to the input data.
    - transform(X, y=None): Transform the input data by expanding the columns.
    """

    def __init__(self, columns_regex):
        self.columns_regex = columns_regex
        super(ExpandColumnPerLevel, self).__init__()

    def fit(self, X, y=None):
        """
        Fit the transformer to the input data.

        Parameters:
        - X (pd.DataFrame): The input data.
        - y (pd.Series or None): The target variable. Not used in this transformer.

        Returns:
        - self (ExpandColumnPerLevel): The fitted transformer object.
        """
        regex_patterns = [re.compile(pattern) for pattern in self.columns_regex]
        self.matched_columns_ = set(
            [
                col
                for col in X.columns
                for pattern in regex_patterns
                if pattern.match(col)
            ]
        )

        self.new_columns_ = {}
        # Ensure identifiers are tuples
        self.series_identifiers_ = [
            idx if isinstance(idx, tuple) else (idx,)
            for idx in X.index.droplevel(-1).unique()
        ]
        for identifier in self.series_identifiers_:
            for col in self.matched_columns_:
                self.new_columns_[self.get_col_name(col, identifier)] = col
        return self

    @classmethod
    def get_col_name(cls, column, identifier):
        """
        Generate the new column name based on the original column name and the identifier.

        Parameters:
        - column (str): The original column name.
        - identifier (tuple): The identifier for the column.

        Returns:
        - new_col_name (str): The new column name.
        """
        return f"{column}_dup_{'-'.join(identifier)}"

    def transform(self, X, y=None):
        """
        Transform the input data by expanding the columns.

        Parameters:
        - X (pd.DataFrame): The input data.
        - y (pd.Series or None): The target variable. Not used in this transformer.

        Returns:
        - X_transformed (pd.DataFrame): The transformed data with expanded columns.
        """
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("Input must have a multi-level index")
        X_original = X.copy()
        X = X.copy()

        regex_patterns = [re.compile(pattern) for pattern in self.columns_regex]
        matched_columns = self.matched_columns_

        # Ensure identifiers are tuples
        series_identifiers = [
            idx if isinstance(idx, tuple) else (idx,)
            for idx in X.index.droplevel(-1).unique()
        ]

        def get_col_name(column, identifier):
            return f"{column}_dup_{'-'.join(identifier)}"

        for match in matched_columns:
            for identifier in series_identifiers:
                X[get_col_name(match, identifier)] = np.float64(0)

        self.new_columns_ = {}
        for identifier in series_identifiers:

            for column in matched_columns:
                self.new_columns_[get_col_name(column, identifier)] = column
                X.loc[identifier, get_col_name(column, identifier)] = X_original.loc[
                    identifier, column
                ].values.flatten()
                X[get_col_name(column, identifier)] = X[
                    get_col_name(column, identifier)
                ].fillna(0)

        return X.drop(columns=matched_columns)
