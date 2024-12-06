from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import (
    TokenTextSplitter,
)
from llama_index.core.schema import BaseNode

from src.database.qdrant import QdrantVectorDB
from ..embedding_model import GeminiEmbedding

import pandas as pd
import os
from dotenv import load_dotenv
import hashlib
import json
import tqdm
import time

load_dotenv()

CHECKPOINT_FILE = "checkpoints.json"
column_names = ["id", 'title', 'abstract', 'authors', 'category', 'year', 'source', 'references']

def load_checkpoints():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}

def generate_hash(doc_id, source):
    return hashlib.sha256(f"{source}_{doc_id}".encode()).hexdigest()

def save_checkpoint(doc_id, source):
    checkpoints = load_checkpoints()
    doc_hash = generate_hash(doc_id, source)
    checkpoints[doc_hash] = True
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoints, f)

def normalize_id(id: str):
    if id.endswith(".0"):
        return id[:-2]
    return id

def initial_offset(csv_path: str):
    checkpoints = load_checkpoints()
    df = pd.read_csv(csv_path)

    filterd_df = df[~df.apply(lambda row: generate_hash(normalize_id(str(row["id"])), row["source"]) in checkpoints, axis=1)].reset_index(drop=True)
    offset = len(df) - len(filterd_df)
    # Clear variable to save memory
    del df
    del filterd_df

    return offset

def load_documents(csv_path: str, offset:int, chunk_size:int) -> List[Document]:
    if not csv_path.endswith(".csv"):
        AssertionError("Please provide a valid csv file")
        return []

    # Load checkpoints
    checkpoints = load_checkpoints()
    df = pd.read_csv(csv_path, skiprows=offset+1, nrows=chunk_size, names=column_names)

    documents = []

    filterd_df = df[~df.apply(lambda row: generate_hash(row["id"], row["source"]) in checkpoints, axis=1)]
    for _, row in filterd_df.iterrows():
        doc = Document(
            text=row["abstract"],
            metadata={
                "id": row["id"],
                "title": row["title"],
                "source": row["source"],
            },
        )
        documents.append(doc)

    return documents

def chunk_documents(documents: List[Document], processed_indices:List[int], chunk_size:int=1024, chunk_overlap:int=256) -> List[BaseNode]:
    text_parser = TokenTextSplitter.from_defaults(chunk_overlap=chunk_overlap,  chunk_size=chunk_size)
    # nodes = text_parser.get_nodes_from_documents(documents, show_progress=True)

    nodes = []
    count_node = 0
    for doc in documents:
        for node in text_parser.get_nodes_from_documents([doc]):
            nodes.append(node)
            count_node += 1
        processed_indices.append(count_node)

    return nodes

def process_chunk(args):
    csv_path, offset, chunk_size, chunk_overlap, lock = args
    documents = load_documents(csv_path=csv_path, offset=offset, chunk_size=chunk_size)
    processed_indices = []
    nodes = chunk_documents(documents, processed_indices, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return nodes, processed_indices

if __name__ == "__main__":

    csv_path = "combined_data.csv"

    document_length = 888907
    init_offset = initial_offset(csv_path=csv_path)
    chunk_size = 20

    COLLECTION_NAME = "DSDE-project-embedding"
    # local_embedding_model = SentenceTransformer("BAAI/bge-m3")
    embedding_model = GeminiEmbedding()

    print(f"Initial offset: {init_offset}")

    qdrantDB = QdrantVectorDB(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), embedding_model=embedding_model, timeout=100, dimension=768)
    qdrantDB.recreate_collection(collection_name=COLLECTION_NAME)


    for offset in tqdm.tqdm(range(init_offset, document_length, chunk_size)):
        start_time = time.time()
        documents = load_documents(csv_path=csv_path, offset=offset, chunk_size=chunk_size)
        processed_indices = []
        nodes = chunk_documents(documents, processed_indices, chunk_size=256, chunk_overlap=100)

        for node in nodes:
            vector = embedding_model.get_embedding(node.get_content())
            print(vector[0][:2])

        # qdrantDB.upload_vectors(collection_name=COLLECTION_NAME, nodes=nodes)
        tmp_nodes = []
        idx = 0
        for i, node in enumerate(nodes):
            tmp_nodes.append(node)

            if i == processed_indices[idx] - 1:
                save_checkpoint(node.metadata["id"], node.metadata["source"])
                idx += 1
                tmp_nodes = []
        end_time = time.time()

        print(f"Chunk processed in {end_time - start_time:.2f} seconds")
        break
 
    qdrantDB.close()
    print("All documents are chunked and uploaded to Qdrant successfully!")