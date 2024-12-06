from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiEmbedding:
    def __init__(self) -> None:
        pass

    def get_embedding(self, documents:List[str]) -> List[List[float]]:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=documents
        )

        return result["embedding"]