from qdrant_client import QdrantClient, models
from llama_index.core.schema import BaseNode
from ..embedding_model import GeminiEmbedding

from typing import List
from dotenv import load_dotenv

from qdrant_client.models import ScoredPoint

from ..interface.vectorDBInterface import VectorDBInterface
from app.models.GraphData import Node, NodeType
import pandas as pd

load_dotenv()

class QdrantVectorDB(VectorDBInterface):
    def __init__(self, url: str, embedding_model: GeminiEmbedding, timeout: int = 100, api_key: str=None,  dimension: int=None):
        """
            A class to interact with a Qdrant vector database, providing functionalities to manage collections,
            upload vectors, and perform searches within the collections.

            Attributes:
                client (QdrantClient): The client instance to communicate with the Qdrant database.
                embedding_model (SetenceTransformer): The embedding model used to convert text to vectors.
        """
        if api_key is None:
            self.client = QdrantClient(url=url, timeout=timeout)
        else:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        self.embedding_model = embedding_model

        if (dimension is None):
            self.dimension = len(self.embedding_model.get_embedding("dummy"))
        else :
            self.dimension = dimension

        self.df = pd.read_csv("combined_data.csv")
        self.df["id"] = self.df["id"].astype(str)


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
        
        existing_collections = self.client.get_collections()
        
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
        # for idx, vector in enumerate(vectors):
        #     print(f"{idx} : {content_list} with {vector[:2]}")
        print(len(content_list))
        print(len(vectors))

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
            query_vector=self.embedding_model.get_embedding([query])[0],
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

    def get_paper_info(self, search_result: List[ScoredPoint]):
        paper_info = {}
        for point in search_result:
            identifier = str(point.payload.get("id")) + '_' + point.payload.get("source")
            if identifier in paper_info:
                continue
            paper_id = str(point.payload.get("id"))
            paper = self.df[self.df["id"] == paper_id].to_dict(orient="records")

            if len(paper) == 0:
                continue

            paper_node = Node(
                id=identifier,
                title=paper[0].get("title"),
                type=NodeType.paper,
                year=int(paper[0].get("year")),
                abstract=paper[0].get("abstract"),
                authors=paper[0].get("authors").split(","),
                source=paper[0].get("source"), 
            )

            paper_info[identifier] = paper_node
        
        return list(paper_info.values())