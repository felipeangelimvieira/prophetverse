import numpyro


def pytest_sessionstart(session):
    
    
    print("Enabling x64")
    # Avoid NaNs in tests
    numpyro.enable_x64()
    
