import pandas as pd
from sktime.transformations.series.fourier import FourierFeatures

def seasonal_transformer(freq="D", yearly_seasonality=False, weekly_seasonality=False) -> pd.DataFrame:
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
        sp_list=sp_list, fourier_terms_list=fourier_term_list, freq=freq
    )
