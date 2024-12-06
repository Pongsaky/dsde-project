from abc import ABC, abstractmethod

class ChatInterface(ABC):
    @abstractmethod
    def __init__(self, url: str, api_key: str, timeout: int):
        pass

    @abstractmethod
    def initial_chat(self, message: str):
        pass

    @abstractmethod
    def continue_chat(self, message: str):
        pass
    
    @abstractmethod
    def clear_chat(self):
        pass

    @abstractmethod
    def get_chat_history(self):
        pass