from hierarchical_prophet.exogenous_priors import get_exogenous_priors, sample_exogenous_coefficients, check_regexes_and_return_remaining_regex
from numpyro import distributions as dist
from jax import numpy as jnp
import pandas as pd
import pytest


def test_empty_dataframe():
    """
    Test with an empty DataFrame.
    """
    X = pd.DataFrame()
    exogenous_priors = {"col1": [(dist.Normal, 0, 1)]}
    default_exogenous_prior = None
    result, permutation_matrix = get_exogenous_priors(
        X, exogenous_priors, default_exogenous_prior
    )

    assert result == []
    assert permutation_matrix.shape == (0,)


def test_none_dataframe():
    """
    Test with None as DataFrame.
    """
    X = None
    exogenous_priors = {"col1": [(dist.Normal, 0, 1)]}
    default_exogenous_prior = None
    result, permutation_matrix = get_exogenous_priors(
        X, exogenous_priors, default_exogenous_prior
    )

    assert result == []
    assert permutation_matrix.shape == (0,)


def test_no_match_columns():
    """
    Test where no columns in X match the regex in exogenous_priors.
    """
    X = pd.DataFrame({"col3": [1, 2], "col4": [3, 4]})
    exogenous_priors = {"col1": (dist.Normal, 0, 1)}
    default_exogenous_prior = (dist.Normal, 10, 1)
    result, permutation_matrix = get_exogenous_priors(
        X, exogenous_priors, default_exogenous_prior
    )

    # Expect the default prior to be used for both columns
    assert len(result) == 1
    assert (
        result[0][0] == "exogenous_coefficients_1"
    )  # Since it appends to the list after no matches
    assert (
        result[0][1].__class__ == dist.Normal
    )
    assert (
        result[0][1].loc == jnp.array([10, 10])
    ).all()
    
    assert (
        result[0][1].scale == jnp.array([1, 1])
    ).all()
    assert (permutation_matrix == jnp.eye(2)).all()


def test_all_columns_matched():
    """
    Test where all columns in X match regex patterns in exogenous_priors.
    """
    X = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    exogenous_priors = {"col[12]": (dist.Normal, 0, 1)}
    default_exogenous_prior = (dist.Normal, 10, 1)
    result, permutation_matrix = get_exogenous_priors(
        X, exogenous_priors, default_exogenous_prior
    )

    assert len(result) == 1
    assert (
        result[0][0].startswith("exogenous_coefficients_")
    )  # Since it appends to the list after no matches
    assert result[0][1].__class__ == dist.Normal
    assert (result[0][1].loc == jnp.array([0, 0])).all()

    assert (result[0][1].scale == jnp.array([1, 1])).all()
    assert (permutation_matrix == jnp.eye(2)).all()


def test_some_columns_matched_with_default_prior():
    """
    Test with some columns in X matched and the rest using the default_exogenous_prior.
    """
    X = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
    exogenous_priors = {"col1": (dist.Normal, 0, 1)}
    default_exogenous_prior = (dist.Normal, 10, 1)
    result, permutation_matrix = get_exogenous_priors(
        X, exogenous_priors, default_exogenous_prior
    )

    # Expect specific match and default prior for the remaining
    assert len(result) == 2
    assert result[0][0] == "exogenous_coefficients_0"
    assert result[1][0] == "exogenous_coefficients_1"  # Default prior applied
    assert permutation_matrix.shape == (3, 3)


def test_with_no_matching_regex():
    """
    Test where no regex in exogenous_priors matches any column in X.
    """
    exogenous_priors = {"no_match_regex": [(dist.Normal, 0, 1)]}
    X = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    # Expected to return regex that matches all columns since no column is set by exogenous priors
    expected = "col1|col2"
    assert check_regexes_and_return_remaining_regex(exogenous_priors, X) == expected


def test_with_all_columns_matched():
    """
    Test where all columns in X are matched and set by regex in exogenous_priors.
    """
    exogenous_priors = {
        "col": [(dist.Normal, 0, 1)]
    }  # Regex "col" will match both "col1" and "col2"
    X = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    # Expected to return an empty string since all columns are set
    expected = ""
    assert check_regexes_and_return_remaining_regex(exogenous_priors, X) == expected


def test_with_some_columns_matched():
    """
    Test with some columns in X matched by regex in exogenous_priors.
    """
    exogenous_priors = {"col1": [(dist.Normal, 0, 1)]}
    X = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    # Expected to return regex that matches the remaining columns
    expected = "col2"
    assert check_regexes_and_return_remaining_regex(exogenous_priors, X) == expected


def test_with_overlapping_columns_error():
    """
    Test with overlapping regex patterns in exogenous_priors, which should raise a ValueError.
    """
    exogenous_priors = {
        "col1": [("Dist", 1)],
        "col": [("Dist", 2)],
    }  # "col" will match both "col1" and "col2", overlapping with "col1"
    X = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    with pytest.raises(ValueError):
        check_regexes_and_return_remaining_regex(exogenous_priors, X)
