"""Configure tests and declare global fixtures."""

import pandas as pd
import pytest

# warnings.filterwarnings("ignore")


def pytest_sessionstart(session):
    """Avoid NaNs in tests."""
    # numpyro.enable_x64()


@pytest.fixture(name="effects_sample_data")
def sample_data():
    """Sample data used at effects tests."""
    return pd.DataFrame(
        {
            "x1": range(10),
            "x2": range(10, 20),
            "log_x1": [0.1 * i for i in range(10)],
            "lin_x2": [0.2 * i for i in range(10, 20)],
        }
    )
