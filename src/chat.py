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

    def call_model(self, state: MessagesState):
        system_template_prompt = self.get_system_template_prompt()
        system_prompt = system_template_prompt.format_messages()

        messages = system_prompt + state["messages"]
        response = self.model.invoke(messages)
        return {"messages": response}

    def initial_chat(self, user_input: UserInput) -> APIResponse:
        user_message = user_input.message
        chat_template_prompt = self.get_chat_template_prompt()

        # TODO Retrieve the paper data from the vectorDB and format it as a message
        search_result = self.qdrant_clinet.get_search_results(self.collection_name, user_message, top_k=8)
        nodes : List[Node] = self.qdrant_clinet.get_paper_info(search_result=search_result)
        paper_data = self.format_nodes_to_text(nodes)
        input_messages = chat_template_prompt.format_messages(paper_data=paper_data)

        chat_id = None
        if user_input.chat_id is not None:
            chat_id = str(uuid.uuid4())
        else:
            chat_id = user_input.chat_id

        config = {"configurable": {"thread_id": chat_id}}

        output = self.app.invoke({"messages": input_messages}, config=config)
        print(output["messages"])

        # TODO Convert JSON output to message and GraphData

        return APIResponse(
            chat_id=chat_id, message=output["messages"][-1], newGraph=None
        )
    
    def test_initial_chat(self, user_input: UserInput):
        user_message = user_input.message
        chat_template_prompt = self.get_chat_template_prompt()

        # TODO Retrieve the paper data from the vectorDB and format it as a message
        search_result = self.qdrant_clinet.get_search_results(self.collection_name, user_message, top_k=8)
        nodes : List[Node] = self.qdrant_clinet.get_paper_info(search_result=search_result)
        paper_data = self.format_nodes_to_text(nodes)
        input_messages = chat_template_prompt.format_messages(paper_data=paper_data)

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

        # TODO Use LLM to detect does it require additional data from vectorDB
        is_additional_data_required = self.detect_additional_data(user_input)

        input_messages = None
        if is_additional_data_required:
            # TODO Retrieve the paper data from the vectorDB and format it as a message
            search_result = self.qdrant_clinet.get_search_results(self.collection_name, user_message, top_k=10)
            nodes : List[Node] = self.qdrant_clinet.get_paper_info(search_result=search_result)
            paper_data = self.format_nodes_to_text(nodes)
            chat_template_prompt = self.get_chat_template_prompt()
            input_messages = chat_template_prompt.format_messages(paper_data=paper_data)
        else :
            chat_template_prompt = self.get_chat_template_prompt()
            input_messages = chat_template_prompt.format_messages()

        output = self.app.invoke({"messages": input_messages}, config=config)

        # TODO Convert JSON output to GraphData

        return APIResponse(
            chat_id=user_input.chat_id, message=output["messages"][-1], newGraph=None
        )
    
    def test_continue_chat(self, user_input: UserInput):
        config = {"configurable": {"thread_id": user_input.chat_id}}
        user_message = user_input.message

        # TODO Use LLM to detect does it require additional data from vectorDB
        is_additional_data_required = self.detect_additional_data(user_input)

        input_messages = None
        if is_additional_data_required:
            # TODO Retrieve the paper data from the vectorDB and format it as a message
            search_result = self.qdrant_clinet.get_search_results(self.collection_name, user_message, top_k=10)
            nodes : List[Node] = self.qdrant_clinet.get_paper_info(search_result=search_result)
            paper_data = self.format_nodes_to_text(nodes)
            chat_template_prompt = self.get_chat_template_prompt()
            input_messages = chat_template_prompt.format_messages(paper_data=paper_data)
        else :
            chat_template_prompt = self.get_chat_template_prompt()
            input_messages = chat_template_prompt.format_messages()

        output = self.app.invoke({"messages": input_messages}, config=config)

        for message in output["messages"]:
            print(message.pretty_print())

        return output["messages"][-1]

    def detect_additional_data(self, user_input: UserInput) -> bool:
        config = {"configurable": {"thread_id": user_input.chat_id}}
        user_message = user_input.message
        current_graph = user_input.currentGraph

        # TODO Use LLM to detect does it require additional data from vectorDB
        detect_additional_data_template = self.get_detect_additional_data_template()
        input_messages = detect_additional_data_template.format_messages(message=user_message, current_graph=current_graph)

        output = self.app.invoke({"messages": input_messages}, config=config)

        ## TODO Check the output message is "Yes" or "No"
        return output["messages"][-1].content == "Yes"
    
    def test_detect_additional_data(self, user_input: UserInput):
        config = {"configurable": {"thread_id": user_input.chat_id}}
        user_message = user_input.message
        current_graph = user_input.currentGraph

        # TODO Use LLM to detect does it require additional data from vectorDB
        detect_additional_data_template = self.get_detect_additional_data_template()
        input_messages = detect_additional_data_template.format_messages(message=user_message, current_graph=current_graph)

        output = self.app.invoke({"messages": input_messages}, config=config)

        for message in output["messages"]:
            print(message.pretty_print())

        return output["messages"][-1]

    def get_chat_history(self, chat_id: str) -> List[str]:
        config = {"configurable": {"thread_id": chat_id}}
        state = self.app.get_state(config=config).values

        return [message for message in state["messages"]]

    def clear_chat(self, chat_id: str):
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

    def format_chat_json_to_graph_link(self, chat_json_response: str) -> List[GraphLink]:
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
        links_data = json.loads(chat_json_response)
        graph_links = [GraphLink(**link) for link in links_data]
        return graph_links