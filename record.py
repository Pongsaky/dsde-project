from typing import List

class Record:
    def __init__(self, title: str, abstract: str, categories: List[str]=None, full_text: str=None, references: List[str]=None, publisher: str = None):
        self.title = title
        self.abstract = abstract
        self.categories = categories
        self.full_text = full_text
        self.references = references
        self.publisher = publisher

    def get_title(self):
        return self.title
    
    def get_abstract(self):
        return self.abstract
    
    def get_categories(self):
        return self.categories
    
    def get_full_text(self):
        return self.full_text
    
    def get_references(self):
        return self.references
    
    def get_publisher(self):
        return self.publisher
    
    def set_title(self, title: str):
        self.title = title

    def set_abstract(self, abstract: str):
        self.abstract = abstract

    def set_categories(self, categories: List[str]):
        self.categories = categories

    def set_full_text(self, full_text: str):
        self.full_text = full_text

    def set_references(self, references: List[str]):
        self.references = references

    def set_publisher(self, publisher: str):
        self.publisher = publisher

    def __str__(self):
        return f"Title: {self.title}\nAbstract: {self.abstract}\nCategories: {self.categories}\nReferences: {self.references}\nPublisher: {self.publisher}"