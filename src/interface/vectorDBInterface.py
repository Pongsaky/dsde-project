from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorDBInterface(ABC):
    """
    Abstract base class for interacting with various database systems, providing a unified interface
    for operations like managing collections, uploading vectors, and performing searches.

    Attributes:
        client: The client instance to communicate with the specific database.
        embedding_model: The embedding model used to convert text to vectors.

    Methods:
        __init__(self, url: str, api_key: str, timeout: int, embedding_model: Any):
            Initializes the DataClient instance with the specified parameters.

        recreate_collection(self, collection_name: str, dimension: int) -> None:
            Abstract method to recreate a collection with a new dimension.

        create_collection(self, collection_name: str, dimension: int) -> None:
            Abstract method to create a new collection based on the specified dimension.

        upload_vectors(self, collection_name: str, nodes: List[Any]) -> None:
            Abstract method to upload vectors to a specified collection.

        get_search_results(self, collection_name: str, query: str, top_k: int = 10) -> List[Any]:
            Abstract method to search for vectors in a collection that are closest to the query vector.

        get_filter_by_metadata(self, collection_name: str, filter: Dict[str, Any], top_k: int = 1000) -> List[Any]:
            Abstract method to filter vectors by metadata in a specified collection.

        get_collection_name(self) -> List[str]:
            Abstract method to retrieve the names of all collections in the database.

        get_client(self) -> Any:
            Abstract method to return the database client instance.

        close(self) -> None:
            Abstract method to close the connection to the database.
    """

    @abstractmethod
    def __init__(self, url: str, api_key: str, timeout: int, embedding_model: Any):
        pass

    @abstractmethod
    def recreate_collection(self, collection_name: str, dimension: int) -> None:
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, dimension: int) -> None:
        pass

    @abstractmethod
    def upload_vectors(self, collection_name: str, nodes: List[Any]) -> None:
        pass

    @abstractmethod
    def get_search_results(self, collection_name: str, query: str, top_k: int = 10) -> List[Any]:
        pass

    @abstractmethod
    def get_filter_by_metadata(self, collection_name: str, filter: Dict[str, Any], top_k: int = 1000) -> List[Any]:
        pass

    @abstractmethod
    def get_collection_name(self) -> List[str]:
        pass