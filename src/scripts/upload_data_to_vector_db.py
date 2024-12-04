from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import (
    TokenTextSplitter,
)
from llama_index.core.schema import BaseNode
from sentence_transformers import SentenceTransformer

from src.database.qdrant import QdrantVectorDB
from src.summarization.summarization import LLMSummarization

import os
from dotenv import load_dotenv

load_dotenv()

def load_documents() -> List[Document]:
    DIR_PATH = "./AFAST_Content"
    SUMMARIZE_PATH = os.path.join(DIR_PATH, "summarize")

    documents = []

    # Switch the split character based on the OS
    split_char = "\\" if os.name == "nt" else "/"

    for root, _, files in os.walk(SUMMARIZE_PATH):
        if files:
            for file in files:
                if file.endswith(".txt"):
                    title = root.split(split_char)[-1]
                    part = file.split(".")[0].split(" ")[-1].replace("EP", "")
                    with open(os.path.join(root, file), "r",  encoding='utf-8') as f:
                        text = f.read()

                    doc = Document(
                        text=text,
                        metadata={
                            "title": title,
                            "part": part,
                            "instructor": "ผศ.ดร.ธรรณพ อารีพรรค",
                            "link": "https://pmdacademy.teachable.com/p/from-nlp-to-llm",
                        },
                    )

                    documents.append(doc)

    return documents

def chunk_documents(documents, chunk_size:int=1024, chunk_overlap:int=512) -> List[BaseNode]:
    text_parser = TokenTextSplitter.from_defaults(chunk_overlap=chunk_overlap, chunk_size=chunk_size)
    nodes = text_parser.get_nodes_from_documents(documents, show_progress=True)

    return nodes

if __name__ == "__main__":
    documents = load_documents()
    nodes = chunk_documents(documents)

    embedding_model = SentenceTransformer(model_name_or_path="model_weight/BAAI_bge_m3")
    qdrantDB = QdrantVectorDB(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API"), embedding_model=embedding_model, timeout=100)

    # Create a new collection with the name and dimension 1024
    COLLECTION_NAME = "AFAST_Summarize_Content"
    EMBEDDING_SIZE = 1024
    qdrantDB.recreate_collection(collection_name=COLLECTION_NAME,)
    qdrantDB.upload_vectors(collection_name=COLLECTION_NAME, nodes=nodes)

    qdrantDB.close()
    print("All documents are chunked and uploaded to Qdrant successfully!")