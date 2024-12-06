from abc import ABC, abstractmethod
from langgraph.graph import MessagesState
from app.models.UserInput import UserInput

class ChatInterface(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def init_app(self):
        pass

    @abstractmethod
    def call_model(self, state: MessagesState):
        pass

    @abstractmethod
    def initial_chat(self, user_input: UserInput):
        pass

    @abstractmethod
    def continue_chat(self, user_input: UserInput):
        pass

    @abstractmethod
    def detect_additional_data(self, user_input: UserInput):
        pass
    
    @abstractmethod
    def clear_chat(self, chat_id:str):
        pass

    @abstractmethod
    def get_chat_history(self, chat_id: str):
        pass