import pandas as pd
from prophetverse.examples.repository.paths import DATASET_PATH, FIG_PATH
from prophetverse.examples.repository.base import ImageRepository, DatasetRepository
import matplotlib.pyplot as plt
from typing import List

___all___ = ['load_dataset', 'list_datasets']

class PandasCSVDatasetRepository(DatasetRepository):
    def read_dataset(self, file_name: str, layer:['raw', 'refined']="refined", **kwargs) -> pd.DataFrame:
        file_path = DATASET_PATH / layer / f"{file_name}.csv"
        return pd.read_csv(file_path, **kwargs)
    
    def list_datasets(self) -> List[str]:
        return [file.stem for file in DATASET_PATH.glob("*.csv")]

    def save_dataset(self, df:pd.DataFrame, file_name: str, **kwargs) -> None:
        layer = "refined"
        file_path = DATASET_PATH / layer / f"{file_name}.csv"
        return df.to_csv(file_path, **kwargs)
    
class MatplotlibImageRepository(ImageRepository):
    
    @staticmethod
    def save_image(file_name: str, fig:plt.Figure)->None:
        file_path = FIG_PATH  / f"{file_name}.png"
        fig.savefig(file_path)

pandas_csv_repo = PandasCSVDatasetRepository()
    
def load_dataset(file_name: str, layer='refined', **kwargs) -> pd.DataFrame:
    return pandas_csv_repo.read_dataset(file_name, layer, **kwargs)

def save_dataset(df: pd.DataFrame, file_name:str, **kwargs):
    pandas_csv_repo.save_dataset(df, file_name, **kwargs)

def list_datasets()->List[str]:
    return pandas_csv_repo.list_datasets()

def save_image(img_name:str, fig:plt.Figure):
    img_repo = MatplotlibImageRepository()
    img_repo.save_image(img_name, fig)