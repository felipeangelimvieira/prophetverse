from prophetverse.examples.preprocess_datasets import Preprocess
from prophetverse.examples.repository.repositories import load_dataset
from prophetverse.examples.repository.repositories import save_image
from prophetverse.examples.plots import unemployment_changepoint

PREPROCESS_RAW = True
GENERATE_FIGS = True

if PREPROCESS_RAW:
    Preprocess().brazilian_unemployment_ibge()

if GENERATE_FIGS:
    fig_unemployment = 
    fig_unemployment_changepoint = unemployment_changepoint()
    save_image("fig_unemployment", fig_unemployment)
