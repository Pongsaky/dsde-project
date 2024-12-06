from email import message
from typing import List
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
from app.models.GraphData import Node

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

        # self.text_parser = StrOutputParser()
        self.system_template_prompt = None
        self.chat_template_prompt = None
        self.detect_additional_data_template = None

        self.workflow = StateGraph(state_schema=MessagesState)
        self.memory_saver = MemorySaver()
        self.app = None

    def init_app(self):
        self.workflow.add_node("model", self.call_model)
        self.workflow.add_edge(START, "model")

        self.app = self.workflow.compile(checkpointer=self.memory_saver)

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
        search_result = self.qdrant_clinet.get_search_results(self.collection_name, user_message, top_k=10)
        nodes : List[Node] = self.qdrant_clinet.get_paper_info(search_result=search_result)
        paper_data = self.format_nodes_to_text(nodes)
        input_messages = chat_template_prompt.format_messages(paper_data=paper_data)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        output = self.app.invoke({"messages": input_messages}, config=config)
        print(output["messages"])

        # TODO Convert JSON output to message and GraphData

        return APIResponse(
            chat_id=thread_id, message=output["messages"][-1], newGraph=None
        )
    
    def test_initial_chat(self, user_input: UserInput):
        user_message = user_input.message
        chat_template_prompt = self.get_chat_template_prompt()

        # TODO Retrieve the paper data from the vectorDB and format it as a message
        search_result = self.qdrant_clinet.get_search_results(self.collection_name, user_message, top_k=10)
        nodes : List[Node] = self.qdrant_clinet.get_paper_info(search_result=search_result)
        paper_data = self.format_nodes_to_text(nodes)
        input_messages = chat_template_prompt.format_messages(paper_data=paper_data)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        output = self.app.invoke({"messages": input_messages}, config=config)
        
        for message in output["messages"]:
            print(message.pretty_print())

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

    def get_chat_history(self, chat_id: str) -> List[str]:
        config = {"configurable": {"thread_id": chat_id}}
        state = self.app.get_state(config=config).values

        return [message for message in state["messages"]]

    def clear_chat(self, chat_id: str):
        pass

    def get_system_template_prompt(self):
        return self.system_template_prompt 

    def set_system_template_prompt(self, text_propmt: str):
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    # """
                    # You are summarization assiatant AI. 
                    # Please help me summarize with the words correctly according to the context by answering in Thai language on {topic} Topic.
                    # Think step by step and be concise.
                    # Don't Forget the RULES:
                    # - Be concise and to the point.
                    # - Length must not exceed 500 characters. 
                    # """
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
            paper_data += f"""ID : {node.id}
Title: {node.title}
Year: {node.year}
Authors: {", ".join(node.authors)}
Source: {node.source}
Abstract: {node.abstract}"""

        return paper_data