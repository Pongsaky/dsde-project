"""
TODO : Add re-process when json response is not valid
TODO : Refactor the code to be more readable
TODO : Separate the code into smaller functions
TODO : Filiter Qdrant Data 
"""

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
from src.interface.ChatInterface import ChatInterface  # Updated import
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
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        self.embedding_model = GeminiEmbedding()
        self.qdrant_client = QdrantVectorDB(url="http://localhost:6333", embedding_model=self.embedding_model)
        self.collection_name = "DSDE-project-embedding"

        self.workflow = StateGraph(state_schema=MessagesState)
        self.workflow.add_node("model", self.call_model)
        self.workflow.add_edge(START, "model")

        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

        self.system_template_prompt = None
        self.chat_template_prompt_text = None
        self.detect_additional_data_template = None

        self.load_templates()

    def load_templates(self):
        templates_path = Path("src/prompt_template")
        self.set_system_template_prompt((templates_path / "system_template_prompt.txt").read_text())
        self.set_chat_template_prompt((templates_path / "chat_template_prompt.txt").read_text())
        self.set_detect_additional_data_template((templates_path / "detect_additional_data_template.txt").read_text())
 
        # Debugging load_templates
        logger.debug("Templates loaded successfully")
        logger.debug("================= System Template Prompt ================")
        logger.debug(self.system_template_prompt)
        logger.debug("================= Chat Template Prompt ================")
        logger.debug(self.chat_template_prompt)
        logger.debug("================= Detect Additional Data Template ================")
        logger.debug(self.detect_additional_data_template)
        

    def call_model(self, state: MessagesState):
        system_template_prompt = self.get_system_template_prompt()
        system_prompt = system_template_prompt.format_messages()

        messages = system_prompt + state["messages"]
        response = self.model.invoke(messages)
        return {"messages": response}

    def initial_chat(self, user_input: UserInput) -> APIResponse:
        user_message = user_input.message
        chat_template_prompt = self.get_chat_template_prompt()

        # Retrieve the paper data from the vectorDB and format it as a message
        paper_data = self._retrieve_and_format_paper_data(user_message)
        logger.info("Retrieving paper data from the vectorDB")

        input_messages = chat_template_prompt.format_messages(paper_data=paper_data, message=user_message)
        logger.info("Paper data retrieved successfully")

        chat_id = user_input.chat_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": chat_id}}
        
        logger.info("Starting initial chat")
        output = self.app.invoke({"messages": input_messages}, config=config)
        logger.info("Initial chat started successfully")

        # Convert JSON output to message and GraphData
        graph_data, summation =  self._format_chat_json_to_graph_link(output["messages"][-1].content)
        logger.info("Chat JSON response formatted successfully")

        return APIResponse(
            chat_id=chat_id, message=summation, newGraph=graph_data
        )
    
    def test_initial_chat(self, user_input: UserInput):
        user_message = user_input.message
        chat_template_prompt = self.get_chat_template_prompt()

        # Retrieve the paper data from the vectorDB and format it as a message
        paper_data = self._retrieve_and_format_paper_data(user_message)
        logger.info("Retrieving paper data from the vectorDB")

        input_messages = chat_template_prompt.format_messages(paper_data=paper_data, message=user_message)
        logger.info("Paper data retrieved successfully")

        chat_id = user_input.chat_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": chat_id}}
        
        logger.info("Starting initial chat")
        output = self.app.invoke({"messages": input_messages}, config=config)
        logger.info("Initial chat started successfully")

        for message in output["messages"]:
            print(message.pretty_print())

        return output["messages"][-1]

    def continue_chat(self, user_input: UserInput) -> APIResponse:
        chat_id = user_input.chat_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": chat_id}}
        user_message = user_input.message

        # Use LLM to detect does it require additional data from vectorDB
        is_additional_data_required = self.detect_additional_data(user_input)
        logger.info("Additional data detected successfully")

        # Retrieve the paper data from the vectorDB and format it as a message
        logger.debug(is_additional_data_required)
        paper_data = self._retrieve_and_format_paper_data(user_message + "\n reason :" + is_additional_data_required.reason) if is_additional_data_required.isNeed else ""
        logger.info("Retrieving paper data from the vectorDB")

        chat_template_prompt = self.get_chat_template_prompt()
        input_messages = chat_template_prompt.format_messages(paper_data=paper_data, message=user_message)
        logger.info("Setting chat template prompt successfully") 

        logger.info("Starting initial chat")
        output = self.app.invoke({"messages": input_messages}, config=config)
        logger.info("Initial chat started successfully")

        logger.info("Formatting chat JSON response to GraphData")     
        graph_data, summation =  self._format_chat_json_to_graph_link(output["messages"][-1].content)
        logger.info("Chat JSON response formatted successfully")

        return APIResponse(
            chat_id=chat_id, message=summation, newGraph=graph_data
        )
    
    def test_continue_chat(self, user_input: UserInput):
        config = {"configurable": {"thread_id": user_input.chat_id}}
        user_message = user_input.message

        # Use LLM to detect does it require additional data from vectorDB
        is_additional_data_required = self.detect_additional_data(user_input)
        logger.info("Additional data detected successfully")

        # Retrieve the paper data from the vectorDB and format it as a message
        logger.debug(is_additional_data_required)
        paper_data = self._retrieve_and_format_paper_data(user_message) if is_additional_data_required.isNeed else ""
        logger.info("Retrieving paper data from the vectorDB")

        chat_template_prompt = self.get_chat_template_prompt()
        input_messages = chat_template_prompt.format_messages(paper_data=paper_data, message=user_message)
        logger.info("Setting chat template prompt successfully") 

        logger.info("Starting initial chat")
        output = self.app.invoke({"messages": input_messages}, config=config)
        logger.info("Initial chat started successfully")

        for message in output["messages"]:
            print(message.pretty_print())

        return output["messages"][-1]

    def detect_additional_data(self, user_input: UserInput) -> DetectAdditionalData:
        config = {"configurable": {"thread_id": user_input.chat_id}}
        user_message = user_input.message
        current_graph = user_input.currentGraph or GraphData(nodes=[], links=[])

        # Use LLM to detect does it require additional data from vectorDB
        detect_additional_data_template = self.get_detect_additional_data_template()
        input_messages = detect_additional_data_template.format_messages(message=user_message, current_graph=current_graph)

        output = self.app.invoke({"messages": input_messages}, config=config)
        chat_json_response = output["messages"][-1].content
        logger.debug("============= Chat JSON Response =============")
        logger.debug(chat_json_response)

        detect_additional_response_json = self._extract_json_from_response(chat_json_response)
        logger.debug("============= Detect Additional Response =============")
        logger.debug(detect_additional_response_json)

        is_need = self._convert_detect_additional_data_to_boolean(detect_additional_response_json["isNeed"])
        reason = detect_additional_response_json["reason"]

        return DetectAdditionalData(isNeed=is_need, reason=reason)
    
    def test_detect_additional_data(self, user_input: UserInput) :
        config = {"configurable": {"thread_id": user_input.chat_id}}
        user_message = user_input.message
        current_graph = user_input.currentGraph

        # Use LLM to detect does it require additional data from vectorDB
        detect_additional_data_template = self.get_detect_additional_data_template()
        input_messages = detect_additional_data_template.format_messages(message=user_message, current_graph=current_graph)

        output = self.app.invoke({"messages": input_messages}, config=config)
        chat_json_response = output["messages"][-1].content
        print(output["messages"][-1].pretty_print())
        logger.debug("============= Chat JSON Response =============")
        logger.debug(chat_json_response)

        detect_additional_response_json = self._extract_json_from_response(chat_json_response)
        logger.debug("============= Detect Additional Response =============")
        logger.debug(detect_additional_response_json)

        is_need = self._convert_detect_additional_data_to_boolean(detect_additional_response_json["isNeed"])
        reason = detect_additional_response_json["reason"]

        return DetectAdditionalData(isNeed=is_need, reason=reason)

    def get_chat_history(self, chat_id: str) -> List[str]:
        config = {"configurable": {"thread_id": chat_id}}
        state = self.app.get_state(config=config).values

        return [message for message in state["messages"]]

    def clear_latest_message(self, chat_id: str):
        config = {"configurable": {"thread_id": chat_id}}
        state = self.app.get_state(config=config).values

        for message in state["messages"][-2:len(state["messages"])]:
            self.app.update_state(config=config, values={"messages": RemoveMessage(id=message.id)})

    def clear_all_message(self, chat_id: str):
        config = {"configurable": {"thread_id": chat_id}}
        state = self.app.get_state(config=config).values

        for message in state["messages"]:
            self.app.update_state(config=config, values={"messages": RemoveMessage(id=message.id)})

    def get_system_template_prompt(self):
        return self.system_template_prompt 

    def set_system_template_prompt(self, text_prompt: str):
        self.system_template_prompt = self._create_chat_prompt_template(SystemMessage(text_prompt))

    def get_chat_template_prompt(self):
        return self.chat_template_prompt

    def set_chat_template_prompt(self, text_prompt: str):
        self.chat_template_prompt_text = text_prompt
        self.chat_template_prompt = self._create_chat_prompt_template(HumanMessagePromptTemplate.from_template(text_prompt))

    def get_detect_additional_data_template(self):
        return self.detect_additional_data_template

    def set_detect_additional_data_template(self, text_prompt: str):
        self.detect_additional_data_template = self._create_chat_prompt_template(HumanMessagePromptTemplate.from_template(text_prompt))

    # Helper functions
    def _retrieve_and_format_paper_data(self, message, top_k=10):
        search_results = self.qdrant_client.get_search_results(self.collection_name, message, top_k)
        filtered_search_results = [search_result for search_result in search_results if search_result.score > 0.6]
        nodes = self.qdrant_client.get_paper_info(search_result=filtered_search_results)
        return self._format_nodes_to_text(nodes)

    def _create_chat_prompt_template(self, message_template):
        return ChatPromptTemplate.from_messages([message_template])
    
    def _format_nodes_to_text(self, nodes: List[Node]) -> str:
        return "\n".join(json.dumps(node.model_dump(), indent=4) for node in nodes)

    def _format_chat_json_to_graph_link(self, chat_json_response: str) -> Tuple[GraphData, str]:
        logger.debug("============= Chat JSON Response =============")
        logger.debug(chat_json_response)

        graph_data = self._extract_json_from_response(chat_json_response)
        nodes_json = graph_data.get("nodes", [])
        links_json = graph_data.get("links", [])

        nodes = [self._create_node(node) for node in nodes_json]
        links = [GraphLink(**link) for link in links_json]

        return GraphData(nodes=nodes or None, links=links or None), graph_data.get("summation", "")

    def _create_node(self, node_data: dict) -> Node:
        node_data.setdefault("year", None)
        node_data.setdefault("abstract", None)
        node_data.setdefault("authors", None)
        node_data.setdefault("source", None)
        if "label" in node_data:
            node_data["title"] = node_data.pop("label", None)
        node_data.setdefault("title", None)
        return Node(**node_data)

    def _convert_detect_additional_data_to_boolean(self, detect_additional_data_response: str) -> bool:
        # normalize the detect_additional_data_response to lowercase and delete \n
        detect_additional_data_response = detect_additional_data_response.lower().replace("\n", "")
        return detect_additional_data_response.lower() == "yes"
    
    def _extract_json_from_response(self, response: str):
        try:
            return json.loads(response.split("```json")[1].split("```")[0])
        except (IndexError, json.JSONDecodeError):
            logger.error("Failed to extract JSON from response")
            return {}