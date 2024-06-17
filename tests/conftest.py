"""Configure tests and declare global fixtures."""

import warnings

import numpyro

warnings.filterwarnings("ignore")


def pytest_sessionstart(session):
    """Avoid NaNs in tests."""
    print("Enabling x64")
    numpyro.enable_x64()
