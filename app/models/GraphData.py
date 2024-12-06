from pydantic import BaseModel
from enum import Enum
from typing import List

class NodeType(str, Enum):
    paper = "paper"
    keyword = "keyword"

class Node(BaseModel):
    id: str
    title: str
    type: NodeType
    year: int
    abstract: str
    authors: List[str]  # Updated field name
    source: str

class GraphNode(BaseModel):
    id: str
    data: Node

class GraphLink(BaseModel):
    source: str
    target: str
    strength: float
    index: int

class GraphData(BaseModel):
    nodes: List[GraphNode]
    links: List[GraphLink]