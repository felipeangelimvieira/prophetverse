import pytest
from prophetverse.examples.preprocess_datasets import Preprocess

@pytest.fixture
def preprocess_instance():
    return Preprocess()

def test_brazilian_unemployment_ibge(preprocess_instance):
    preprocess_instance.brazilian_unemployment_ibge(save=False)