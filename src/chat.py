from typing import List
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
        self.qdrant_clinet = QdrantVectorDB(url="http://localhost:6333", embedding_model=self.embedding_model)
        self.collection_name = "DSDE-project-embedding"

        self.workflow = StateGraph(state_schema=MessagesState)
        self.workflow.add_node("model", self.call_model)
        self.workflow.add_edge(START, "model")

        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

        self.chat_template_prompt_text = None

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
        search_result = self.qdrant_clinet.get_search_results(self.collection_name, user_message, top_k=8)
        nodes : List[Node] = self.qdrant_clinet.get_paper_info(search_result=search_result)
        paper_data = self.format_nodes_to_text(nodes)
        input_messages = chat_template_prompt.format_messages(paper_data=paper_data, message=user_message)

        chat_id = None
        if user_input.chat_id is not None:
            chat_id = str(uuid.uuid4())
        else:
            chat_id = user_input.chat_id

        config = {"configurable": {"thread_id": chat_id}}

        output = self.app.invoke({"messages": input_messages}, config=config)
        print(output["messages"])

        # Convert JSON output to message and GraphData
        graph_data, summation =  self.format_chat_json_to_graph_link(output["messages"][-1].content)

        return APIResponse(
            chat_id=chat_id, message=summation, newGraph=graph_data
        )
    
    def test_initial_chat(self, user_input: UserInput):
        user_message = user_input.message
        chat_template_prompt = self.get_chat_template_prompt()

        # Retrieve the paper data from the vectorDB and format it as a message
        search_result = self.qdrant_clinet.get_search_results(self.collection_name, user_message, top_k=8)
        nodes : List[Node] = self.qdrant_clinet.get_paper_info(search_result=search_result)
        paper_data = self.format_nodes_to_text(nodes)
        input_messages = chat_template_prompt.format_messages(paper_data=paper_data, message=user_message)

        chat_id = None
        if user_input.chat_id is not None:
            chat_id = user_input.chat_id
        else:
            chat_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": chat_id}}

        output = self.app.invoke({"messages": input_messages}, config=config)
        
        for message in output["messages"]:
            print(message.pretty_print())

        return output["messages"][-1]

    def continue_chat(self, user_input: UserInput) -> APIResponse:
        config = {"configurable": {"thread_id": user_input.chat_id}}
        user_message = user_input.message

        # Use LLM to detect does it require additional data from vectorDB
        is_additional_data_required = self.detect_additional_data(user_input)

        input_messages = None
        if is_additional_data_required.isNeed:
            # Retrieve the paper data from the vectorDB and format it as a message
            search_result = self.qdrant_clinet.get_search_results(self.collection_name, user_message, top_k=10)
            nodes : List[Node] = self.qdrant_clinet.get_paper_info(search_result=search_result)
            paper_data = self.format_nodes_to_text(nodes)
            chat_template_prompt = self.get_chat_template_prompt()
            input_messages = chat_template_prompt.format_messages(paper_data=paper_data, message=user_message)
        else :
            self.set_chat_template_prompt(self.chat_template_prompt_text.replace("{paper_data}", ""))
            chat_template_prompt = self.get_chat_template_prompt()
            input_messages = chat_template_prompt.format_messages(message=user_message)

        output = self.app.invoke({"messages": input_messages}, config=config)

        if self.chat_template_prompt_text.find("{paper_data}") == -1:
            chat_template_text = "".join([message.content for message in chat_template_prompt.format_messages(message=user_message)]) + "\n{paper_data}"
            self.set_chat_template_prompt(chat_template_text)

        # Convert JSON output to GraphData
        graph_data, summation =  self.format_chat_json_to_graph_link(output["messages"][-1].content)

        return APIResponse(
            chat_id=user_input.chat_id, message=summation, newGraph=graph_data
        )
    
    def test_continue_chat(self, user_input: UserInput):
        config = {"configurable": {"thread_id": user_input.chat_id}}
        user_message = user_input.message

        # Use LLM to detect does it require additional data from vectorDB
        is_additional_data_required = self.detect_additional_data(user_input)
        print(is_additional_data_required)

        input_messages = None
        if is_additional_data_required.isNeed:
            # Retrieve the paper data from the vectorDB and format it as a message
            search_result = self.qdrant_clinet.get_search_results(self.collection_name, user_message, top_k=10)
            nodes : List[Node] = self.qdrant_clinet.get_paper_info(search_result=search_result)
            paper_data = self.format_nodes_to_text(nodes)
            chat_template_prompt = self.get_chat_template_prompt()
            input_messages = chat_template_prompt.format_messages(paper_data=paper_data, message=user_message)
        else :
            self.set_chat_template_prompt(self.chat_template_prompt_text.replace("{paper_data}", ""))
            chat_template_prompt = self.get_chat_template_prompt()
            input_messages = chat_template_prompt.format_messages(message=user_message)

        output = self.app.invoke({"messages": input_messages}, config=config)

        if self.chat_template_prompt_text.find("{paper_data}") == -1:
            chat_template_text = "".join([message.content for message in chat_template_prompt.format_messages(message=user_message)]) + "\n{paper_data}"
            self.set_chat_template_prompt(chat_template_text)

        for message in output["messages"]:
            print(message.pretty_print())

        # TODO Convert JSON output to GraphData

        return output["messages"][-1]

    def detect_additional_data(self, user_input: UserInput) -> DetectAdditionalData:
        config = {"configurable": {"thread_id": user_input.chat_id}}
        user_message = user_input.message
        current_graph = user_input.currentGraph

        # Use LLM to detect does it require additional data from vectorDB
        detect_additional_data_template = self.get_detect_additional_data_template()
        input_messages = detect_additional_data_template.format_messages(message=user_message, current_graph=current_graph)

        output = self.app.invoke({"messages": input_messages}, config=config)
        chat_json_response = output["messages"][-1].content
        print(output["messages"][-1].content)

        chat_json_response = chat_json_response.split("```json")[1].split("```")[0]
        detect_additional_response_json = json.loads(chat_json_response)

        detect_additional_response = DetectAdditionalData(
            isNeed=self.convert_detect_additional_data_to_boolean(detect_additional_response_json["isNeed"]),
            reason=detect_additional_response_json["reason"]
        )

        return detect_additional_response
    
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

        chat_json_response = chat_json_response.split("```json")[1].split("```")[0]
        detect_additional_response_json = json.loads(chat_json_response)

        detect_additional_response = DetectAdditionalData(
            isNeed=self.convert_detect_additional_data_to_boolean(detect_additional_response_json["isNeed"]),
            reason=detect_additional_response_json["reason"]
        )

        return detect_additional_response

    def get_chat_history(self, chat_id: str) -> List[str]:
        config = {"configurable": {"thread_id": chat_id}}
        state = self.app.get_state(config=config).values

        return [message for message in state["messages"]]

    def clear_latest_message(self, chat_id: str):
        config = {"configurable": {"thread_id": chat_id}}
        state = self.app.get_state(config=config).values
        latest_message = state["messages"][-1]
        self.app.update_state(config=config, values={"messages": RemoveMessage(id=latest_message.id)})

    def clear_all_message(self, chat_id: str):
        config = {"configurable": {"thread_id": chat_id}}

        state = self.app.get_state(config=config).values
        for message in state["messages"]:
            self.app.update_state(config=config, values={"messages": RemoveMessage(id=message.id)})

    def get_system_template_prompt(self):
        return self.system_template_prompt 

    def set_system_template_prompt(self, text_propmt: str):
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    text_propmt
                )
            ]
        )

        self.system_template_prompt = chat_prompt_template

    def get_chat_template_prompt(self):
        return self.chat_template_prompt

    def set_chat_template_prompt(self, text_prompt: str):
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    text_prompt
                )
            ]
        )

        self.chat_template_prompt_text = text_prompt
        self.chat_template_prompt = chat_prompt_template

    def get_detect_additional_data_template(self):
        return self.detect_additional_data_template

    def set_detect_additional_data_template(self, text_prompt: str): 
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    text_prompt
                )
            ]
        )

        self.detect_additional_data_template = chat_prompt_template 
    def get_model(self):
        return self.model

    def format_nodes_to_text(self, nodes: List[Node]) -> str:
        paper_data = ""
        for node in nodes:
            node_json = json.dumps(node.model_dump(), indent=4)
            paper_data += f"{node_json}\n"
        return paper_data

    def format_chat_json_to_graph_link(chat_json_response: str) -> List[GraphLink]:
        """
        Example of chat_json_response
        ```json
        [
            {
                "source": "some_id",
                "target": "some_id",
                "index": 0
            }
        ]
        ```
        """

        # Select content between ```json and ```
        chat_json_response = chat_json_response.split("```json")[1].split("```")[0]
        # print(chat_json_response)
        graph_data = json.loads(chat_json_response)
        # print(graph_data)
        nodes_json = graph_data["nodes"]
        links_json = graph_data["links"]

        nodes = []
        for node in nodes_json:
            if "label" in node:
                node["title"] = node["label"]
                del node["label"]
            
            if "year" not in node:
                node["year"] = None
            
            if "abstract" not in node:
                node["abstract"] = None

            if "authors" not in node:
                node["authors"] = None

            if "source" not in node:
                node["source"] = None
                
            node = Node(**node)
            nodes.append(node)

        links = []
        for link in links_json:
            link = GraphLink(**link)
            links.append(link)

        return GraphData(nodes=nodes, links=links), graph_data["summation"]
    
    def convert_detect_additional_data_to_boolean(self, detect_additional_data_response: str) -> bool:
        # normalize the detect_additional_data_response to lowercase and delete \n
        detect_additional_data_response = detect_additional_data_response.lower().replace("\n", "")
        return detect_additional_data_response.lower() == "yes"