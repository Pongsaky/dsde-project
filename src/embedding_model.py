from typing import List
from openai import AzureOpenAI
from langchain_google_vertexai import VertexAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class OpenAIEmbedding:
    def __init__(self, host, api_key) -> None:
        self.host = host
        self.api_key = api_key
        self.client = AzureOpenAI(
            azure_endpoint=self.host,
            api_key=self.api_key,
            api_version="2024-02-01",
        )

    def get_embedding(self, text: str):
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
            dimensions=1024,
        )

        return response.data[0].embedding
    
class GeminiEmbedding:
    def __init__(self) -> None:
        self.embedding_model = VertexAIEmbeddings(model="text-embedding-004")

    def get_embedding(self, documents:List[str]):
        embedding = self.embedding_model.embed_documents(documents)
        return embedding