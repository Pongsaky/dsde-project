from typing import List, Tuple
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from app.models.APIResponse import APIResponse
from app.models.UserInput import UserInput
from app.models.DetectAdditionalData import DetectAdditionalData

from dotenv import load_dotenv
from src.interface.ChatInterface import ChatInterface
import uuid

from src.database.qdrant import QdrantVectorDB
from src.embedding_model import GeminiEmbedding
from app.models.GraphData import Node, GraphLink, GraphData
import json

from langchain_core.messages import RemoveMessage
from pathlib import Path
from app.logger import logger

load_dotenv()

class Chat(ChatInterface):
    def __init__(self):
        self.model = self._initialize_model()
        self.embedding_model = GeminiEmbedding()
        self.qdrant_client = QdrantVectorDB(url="http://localhost:6333", embedding_model=self.embedding_model)
        self.collection_name = "DSDE-project-embedding"

        self.workflow, self.memory = self._initialize_workflow()
        self.system_template_prompt, self.chat_template_prompt_text, self.detect_additional_data_template = None, None, None

        self._load_templates()

    def _initialize_model(self):
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )

    def _initialize_workflow(self):
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("model", self.call_model)
        workflow.add_edge(START, "model")
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory), memory

    def _load_templates(self):
        templates_path = Path("src/prompt_template")
        self.system_template_prompt = self._load_template(templates_path / "system_template_prompt.txt")
        self.chat_template_prompt_text = self._load_template(templates_path / "chat_template_prompt.txt")
        self.detect_additional_data_template = self._load_template(templates_path / "detect_additional_data_template.txt")
        logger.debug("Templates loaded successfully")

    @staticmethod
    def _load_template(path):
        try:
            return path.read_text()
        except FileNotFoundError:
            logger.error(f"Template not found: {path}")
            return ""

    def call_model(self, state: MessagesState):
        system_prompt = self.system_template_prompt.format_messages()
        response = self.model.invoke(system_prompt + state["messages"])
        return {"messages": response}

    def initial_chat(self, user_input: UserInput) -> APIResponse:
        chat_id = user_input.chat_id or str(uuid.uuid4())
        paper_data = self._retrieve_and_format_paper_data(user_input.message)
        input_messages = self.chat_template_prompt_text.format_messages(paper_data=paper_data, message=user_input.message)

        config = {"configurable": {"thread_id": chat_id}}
        output = self.app.invoke({"messages": input_messages}, config=config)

        graph_data, summation = self._parse_chat_response(output["messages"][-1].content)
        return APIResponse(chat_id=chat_id, message=summation, newGraph=graph_data)

    def test_initial_chat(self, user_input: UserInput):
        chat_id = user_input.chat_id or str(uuid.uuid4())
        paper_data = self._retrieve_and_format_paper_data(user_input.message)
        input_messages = self.chat_template_prompt_text.format_messages(paper_data=paper_data, message=user_input.message) 

        config = {"configurable": {"thread_id": chat_id}}
        output = self.app.invoke({"messages": input_messages}, config=config)

        for message in output["messages"]:
            print(message.content.pretty_print( ))

        return output["messages"][-1]

    def continue_chat(self, user_input: UserInput) -> APIResponse:
        chat_id = user_input.chat_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": chat_id}}
        is_additional_data_required = self.detect_additional_data(user_input)

        if is_additional_data_required.isNeed:
            paper_data = self._retrieve_and_format_paper_data(user_input.message, top_k=10)
        else:
            paper_data = ""

        input_messages = self.chat_template_prompt_text.format_messages(paper_data=paper_data, message=user_input.message)
        output = self.app.invoke({"messages": input_messages}, config=config)

        graph_data, summation = self._parse_chat_response(output["messages"][-1].content)
        return APIResponse(chat_id=chat_id, message=summation, newGraph=graph_data)
    
    def test_continue_chat(self, user_input: UserInput):
        config = {"configurable": {"thread_id": user_input.chat_id}}
        is_additional_data_required = self.detect_additional_data(user_input)

        if is_additional_data_required.isNeed:
            paper_data = self._retrieve_and_format_paper_data(user_input.message, top_k=10)
        else:
            paper_data = ""

        input_messages = self.chat_template_prompt_text.format_messages(paper_data=paper_data, message=user_input.message)
        output = self.app.invoke({"messages": input_messages}, config=config)

        for message in output["messages"]:
            print(message.content.pretty_print( ))

        return output["messages"][-1]

    def detect_additional_data(self, user_input: UserInput) -> DetectAdditionalData:
        detect_template = self.detect_additional_data_template.format_messages(
            message=user_input.message, current_graph=user_input.currentGraph
        )

        output = self.app.invoke({"messages": detect_template}, config={"configurable": {"thread_id": user_input.chat_id}})
        try:
            chat_json_response = self._extract_json_from_response(output["messages"][-1].content)
            return DetectAdditionalData(
                isNeed=bool(chat_json_response["isNeed"]), reason=chat_json_response["reason"]
            )
        except (KeyError, json.JSONDecodeError):
            logger.error("Invalid JSON response from model")
            return DetectAdditionalData(isNeed=False, reason="Invalid response")
        
    def test_detect_additional_data(self, user_input: UserInput):
        detect_template = self.detect_additional_data_template.format_messages(
            message=user_input.message, current_graph=user_input.currentGraph
        )

        output = self.app.invoke({"messages": detect_template}, config={"configurable": {"thread_id": user_input.chat_id}})
        for message in output["messages"]:
            print(message.content.pretty_print())

        return output["messages"][-1]

    def _retrieve_and_format_paper_data(self, message, top_k=8):
        search_results = self.qdrant_client.get_search_results(self.collection_name, message, top_k)
        nodes = self.qdrant_client.get_paper_info(search_result=search_results)
        return self._format_nodes_to_text(nodes)

    @staticmethod
    def _extract_json_from_response(response):
        try:
            return json.loads(response.split("```json")[1].split("```")[0])
        except (IndexError, json.JSONDecodeError):
            logger.error("Failed to extract JSON from response")
            return {}

    @staticmethod
    def _parse_chat_response(response):
        try:
            graph_json = json.loads(response)
            graph_data = GraphData(
                nodes=[Node(**node) for node in graph_json.get("nodes", [])],
                links=[GraphLink(**link) for link in graph_json.get("links", [])],
            )
            summation = graph_json.get("summary", "")
            return graph_data, summation
        except json.JSONDecodeError:
            logger.error("Failed to parse chat response")
            return GraphData(nodes=[], links=[]), "Error in processing"

    @staticmethod
    def _format_nodes_to_text(nodes: List[Node]) -> str:
        return "\n".join([f"{node.title}: {node.abstract}" for node in nodes])