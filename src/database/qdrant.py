from qdrant_client import QdrantClient, models
from llama_index.core.schema import BaseNode
from ..embedding_model import GeminiEmbedding

from typing import List
from dotenv import load_dotenv

from qdrant_client.models import ScoredPoint

from ..interface.vectorDBInterface import VectorDBInterface

load_dotenv()

class QdrantVectorDB(VectorDBInterface):
    def __init__(self, url: str, api_key: str, embedding_model: GeminiEmbedding, timeout: int = 100,  dimension: int=None):
        """
            A class to interact with a Qdrant vector database, providing functionalities to manage collections,
            upload vectors, and perform searches within the collections.

            Attributes:
                client (QdrantClient): The client instance to communicate with the Qdrant database.
                embedding_model (SetenceTransformer): The embedding model used to convert text to vectors.
        """
        self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        self.embedding_model = embedding_model

        if (dimension is None):
            self.dimension = len(self.embedding_model.get_embedding("dummy"))
        else :
            self.dimension = dimension


    def recreate_collection(self, collection_name: str) -> None:
        """
            Recreates a collection with a new dimension. This is intended for development purposes.
        """
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def create_collection(self, collection_name: str) -> None:
        """
            Creates a new collection if it does not already exist, based on the specified dimension.
        """
        
        existing_collections = self.client.list_collections()
        
        if collection_name in existing_collections:
            print(f"Collection '{collection_name}' already exists. Skipping recreation.")
            return
        
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE,
            ),
        )
    
    
    def upload_vectors(self, collection_name: str, nodes: List[BaseNode]) -> None:
        """
            Uploads vectors to a specified collection.
        """
        content_list = []
        for node in nodes:
            content_list.append(node.get_content())

        vectors = self.embedding_model.get_embedding(content_list)
         
        self.client.upload_points(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=idx,
                    vector=vectors[idx],
                    payload=dict(node.metadata, **{"content": node.get_content()}),
                )
                for idx, node in enumerate(nodes)
            ],
        )

    def get_search_results(self, collection_name: str, query: str, top_k: int = 10) -> List[ScoredPoint]:
        """
            Searches for vectors in a collection that are closest to the query vector.
        """
        data = self.client.search(
            collection_name=collection_name,
            query_vector=self.embedding_model.embedding_model([query]),
            limit=top_k,
        )

        return data
    
    def get_filter_by_metadata(self, collection_name: str, filter: dict, top_k: int = 1000):
        """
            Filters vectors by metadata in a specified collection.
        """
        data = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )

                    for key, value in filter.items()
                ]
            ),
            limit=top_k
        )

        return data[0]
    
    def get_collection_name(self):
        """
            Retrieves the names of all collections in the database.
        """
        return self.client.get_collections()
    
    def get_client(self):
        """
            Returns the Qdrant client instance.
        """
        return self.client

    def close(self):
        """
            Closes the connection to the Qdrant database.
        """
        self.client.close()