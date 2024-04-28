from hierarchical_prophet.utils.multiindex import reindex_time_series
import pandas as pd
import numpy as np

def test_reindex_timeseries():

    df = pd.DataFrame(
        data=np.random.randn(10, 2),
        index=pd.MultiIndex.from_product(
            [["A", "B"], pd.date_range(start="2021-01-01", periods=5, freq="D")],
            names=["group" , "time"],
        ),
        columns=["value1", "value2"],
    )
    
    output = reindex_time_series(df, pd.date_range(start="2021-01-01", periods=10, freq="D"))
    
    assert output.index.equals(pd.MultiIndex.from_product(
        [["A", "B"], pd.date_range(start="2021-01-01", periods=10, freq="D")],
        names=["group" , "time"],
    ))
