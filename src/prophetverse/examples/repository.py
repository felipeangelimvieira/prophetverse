from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from pathlib import Path
from typing import List, Literal

___all___ = ['load_dataset', 'list_datasets']

DATASET_PATH = Path(__file__).parent / "datasets"

class CSVRepository(ABC):
    @abstractmethod
    def read_csv(self, file_name: str, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def list_datasets(self) -> List[str]:
        pass

class PandasCSVDatasetRepository(CSVRepository):
    def read_csv(self, file_name: str, layer:['raw', 'refined'], **kwargs) -> pd.DataFrame:
        file_path = DATASET_PATH / layer / f"{file_name}.csv"
        return pd.read_csv(file_path, **kwargs)
    
    def list_datasets(self) -> List[str]:
        return [file.stem for file in DATASET_PATH.glob("*.csv")]


def load_dataset(file_name: str, layer, **kwargs) -> pd.DataFrame:
    pandas_csv_repo = PandasCSVDatasetRepository()
    return pandas_csv_repo.read_csv(file_name, layer, **kwargs)


def list_datasets():
    pandas_csv_repo = PandasCSVDatasetRepository()
    return pandas_csv_repo.list_datasets()

