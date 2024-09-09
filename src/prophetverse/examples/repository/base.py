from abc import ABC, abstractmethod
from typing import Any
from typing import List

class ImageRepository(ABC):
    @abstractmethod
    def save_image(self,file_name:str)->None:
        pass
    
class DatasetRepository(ABC):
    @abstractmethod
    def read_dataset(self, file_name: str, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def list_datasets(self) -> List[str]:
        pass
    
    @abstractmethod
    def save_dataset(self, df:Any, file_name: str, **kwargs) -> None:
        pass
