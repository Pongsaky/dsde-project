from abc import ABC, abstractmethod
from app.models.UserInput import UserInput

class ChatInterface(ABC):
    @abstractmethod
    def __init__(self):
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
    def clear_all_message(self, chat_id:str):
        pass

    @abstractmethod
    def get_chat_history(self, chat_id: str):
        pass