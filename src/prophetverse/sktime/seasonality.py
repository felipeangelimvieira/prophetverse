"""Utility to obtain the seasonal transformer for sktime."""

import pandas as pd
from sktime.transformations.series.fourier import FourierFeatures

from prophetverse.utils.deprecation import deprecation_warning


# TODO(felipeangelimvieira): Remove this function in version 0.6.0
def seasonal_transformer(
    freq="D", yearly_seasonality=False, weekly_seasonality=False
) -> pd.DataFrame:
    """
    Transform the seasonality parameters into a pandas DataFrame.

    Parameters
    ----------
    freq : str, optional
        The frequency of the time series. Defaults to "D".
    yearly_seasonality : bool or int, optional
        Whether to include yearly seasonality. If bool, the number of Fourier terms is
        set to 10. If int, the number of Fourier terms is set to the specified value.
        Defaults to False.
    weekly_seasonality : bool or int, optional
        Whether to include weekly seasonality. If bool, the number of Fourier terms
        is set to 3. If int, the number of Fourier terms is set to the specified value.
        Defaults to False.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the seasonality parameters.

    Raises
    ------
    ValueError
        If yearly_seasonality or weekly_seasonality is not a boolean or an integer.
    """
    deprecation_warning(
        "seasonal_transformer",
        "0.5.0",
        "Please use prophetverse.effects.LinearFourierSeasonality instead",
    )
    sp_list = []
    fourier_term_list = []

    if isinstance(yearly_seasonality, bool):
        yearly_seasonality_num_terms = 10
    elif isinstance(yearly_seasonality, int):
        yearly_seasonality_num_terms = yearly_seasonality
    else:
        raise ValueError("yearly_seasonality must be a boolean or an integer")
    if yearly_seasonality:
        sp_list.append("Y")
        fourier_term_list.append(yearly_seasonality_num_terms)

    if isinstance(weekly_seasonality, bool):
        weekly_seasonality_num_terms = 3
    elif isinstance(weekly_seasonality, int):
        weekly_seasonality_num_terms = weekly_seasonality
    else:
        raise ValueError("weekly_seasonality must be a boolean or an integer")

    if weekly_seasonality:
        sp_list.append("W")
        fourier_term_list.append(weekly_seasonality_num_terms)

    return FourierFeatures(
        sp_list=sp_list,
        fourier_terms_list=fourier_term_list,
        freq=freq,
        keep_original_columns=True,
    )
