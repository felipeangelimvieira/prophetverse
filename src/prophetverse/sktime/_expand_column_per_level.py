import re
from typing import Optional

import numpy as np
import pandas as pd
from sktime.transformations.base import BaseTransformer


class ExpandColumnPerLevel(BaseTransformer):
    """
    Expand columns based on a regular expression pattern per level of a multi-level idx.

    If there are M columns matching the pattern and N distinct multi-level indexes, then
    the transformer will create M * N new columns, setting the value to zero to each of
    M * (N-1) columns for a given index j if the current index i /= j.

    Parameters
    ----------
    columns_regex : list
        A list of regular expression patterns to match the columns to be expanded.

    Attributes
    ----------
    matched_columns_ : set
        A set of columns that match the regular expression patterns.
    new_columns_ : dict
        A dictionary mapping the new column names to the original column names.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.
    transform(X, y=None)
        Transform the input data by expanding the columns.
    """

    def __init__(self, columns_regex: list[str]):
        self.columns_regex = columns_regex
        super().__init__()

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.Series or None
            The target variable. Not used in this transformer.

        Returns
        -------
        self : ExpandColumnPerLevel
            The fitted transformer object.
        """
        regex_patterns = [re.compile(str(pattern)) for pattern in self.columns_regex]
        self.matched_columns_ = {
            col
            for col in X.columns
            for pattern in regex_patterns
            if pattern.match(str(col))
        }

        self.new_columns_ = {}
        # Ensure identifiers are tuples
        if X.index.nlevels == 1:
            self.skip_transform_ = True
            self.series_identifiers_ = None
            self.new_columns_ = {col: col for col in self.matched_columns_}
            return self

        self.skip_transform_ = False
        self.series_identifiers_ = [
            idx if isinstance(idx, tuple) else (idx,)
            for idx in X.index.droplevel(-1).unique()
        ]
        for identifier in self.series_identifiers_:
            for col in self.matched_columns_:
                self.new_columns_[self.get_col_name(col, identifier)] = col
        return self

    @classmethod
    def get_col_name(cls, column: str, identifier: tuple) -> str:
        """
        Generate the new column name based on the original name and the identifier.

        Parameters
        ----------
        column : str
            The original column name.
        identifier : tuple
            The identifier for the column.

        Returns
        -------
        new_col_name : str
            The new column name.
        """
        return f"{column}_dup_{'-'.join(identifier)}"

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """
        Transform the input data by expanding the columns.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.Series or None
            The target variable. Not used in this transformer.

        Returns
        -------
        X_transformed : pd.DataFrame
            The transformed data with expanded columns.
        """
        if self.skip_transform_:
            return X

        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("Input must have a multi-level index")
        X_original = X.copy()
        X = X.copy()

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

    @classmethod
    def get_test_params(cls, parameter_set="default"):

        return [
            {"columns_regex": ["*"]},
        ]
