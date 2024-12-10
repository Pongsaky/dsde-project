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
import logging

load_dotenv()

CHECKPOINT_FILE = "checkpoints.json"
column_names = ["id", 'title', 'abstract', 'authors', 'category', 'year', 'source', 'references']

# Configure logging
logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_checkpoints():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_checkpoints(checkpoints):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoints, f) 
    logging.info("Checkpoints saved to file")

def generate_hash(doc_id, source):
    return hashlib.sha256(f"{source}_{doc_id}".encode()).hexdigest()

def save_checkpoint(doc_id, source, checkpoints):
    doc_hash = generate_hash(doc_id, source)
    checkpoints[doc_hash] = True
    logging.info(f"Checkpoint saved for document ID: {doc_id}, source: {source}")

def normalize_id(id: str):
    if id.endswith(".0"):
        return id[:-2]
    return id

def initial_offset(df: pd.DataFrame):
    checkpoints = load_checkpoints()

    filterd_df = df[~df.apply(lambda row: generate_hash(normalize_id(str(row["id"])), row["source"]) in checkpoints, axis=1)].reset_index(drop=True)
    offset = len(df) - len(filterd_df)
    # Clear variable to save memory
    del df
    del filterd_df

    return offset

def load_documents(df : pd.DataFrame, offset:int, chunk_size:int, checkpoints) -> List[Document]:
    df = df.iloc[offset:offset+chunk_size]

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

def process_chunk(offset, chunk_size, df, checkpoints, collection_name):
    logging.info(f"Processing chunk with offset: {offset}")
    
    # Initialize QdrantVectorDB inside the worker function
    documents = load_documents(df=df, offset=offset, chunk_size=chunk_size, checkpoints=checkpoints)
    processed_indices = []
    nodes = chunk_documents(documents, processed_indices, chunk_size=512, chunk_overlap=128)
    logging.info(f"Chunked {len(nodes)} nodes")
    
    qdrantDB.upload_vectors(collection_name=collection_name, nodes=nodes)
    logging.info(f"Uploaded {len(nodes)} nodes to Qdrant")

    idx = 0
    for i, node in enumerate(nodes):
        if i == processed_indices[idx] - 1:
            save_checkpoint(node.metadata["id"], node.metadata["source"], checkpoints)
            idx += 1

    del documents
    del nodes
    logging.info(f"Finished processing chunk with offset: {offset}")

if __name__ == "__main__":

    csv_path = "combined_data.csv"

    document_length = 888907
    df = pd.read_csv(csv_path)
    init_offset = initial_offset(df=df)
    chunk_size = 888907

    COLLECTION_NAME = "DSDE-project-embedding"
    embedding_model = GeminiEmbedding()

    print(f"Initial offset: {init_offset}")
    logging.info(f"Initial offset: {init_offset}")

    # qdrantDB = QdrantVectorDB(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), embedding_model=embedding_model, timeout=100)
    qdrantDB = QdrantVectorDB(url="http://localhost:6333", embedding_model=embedding_model, timeout=100)
    qdrantDB.recreate_collection(collection_name=COLLECTION_NAME)
    logging.info("Qdrant collection created")

    checkpoints = load_checkpoints()

    for offset in tqdm.tqdm(range(init_offset, document_length, chunk_size)):
        start_time = time.time()
        process_chunk(offset, chunk_size, df, checkpoints, COLLECTION_NAME)
        end_time = time.time()
        logging.info(f"Chunk processed in {end_time - start_time:.2f} seconds")

    save_checkpoints(checkpoints)
    qdrantDB.close()

    logging.info("All documents are chunked and uploaded to Qdrant successfully!")
    print("All documents are chunked and uploaded to Qdrant successfully!")