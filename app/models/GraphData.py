from pydantic import BaseModel
from enum import Enum
from typing import List, Optional

class NodeType(str, Enum):
    paper = "paper"
    keyword = "keyword"

class Node(BaseModel):
    id: str
    title: str
    type: NodeType
    year: Optional[int]
    abstract: Optional[str]
    authors: Optional[List[str]]
    source: Optional[str]

class GraphLink(BaseModel):
    source: str
    target: str

class GraphData(BaseModel): 
    nodes: List[Node]
    links: List[GraphLink]