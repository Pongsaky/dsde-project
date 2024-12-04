from typing import List
from langchain_openai.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from llama_index.core.text_splitter import TokenTextSplitter
from dotenv import load_dotenv


load_dotenv()

class LLMSummarization:
    def __init__(self, model:str, api_key:str, base_url:str, max_tokens:int = None, temperature:float = None, isAzure:bool = False):

        if isAzure is False:
            if max_tokens is None:
                raise ValueError("max_tokens must be provided when isAzure is False")
            if temperature is None:
                raise ValueError("temperature must be provided when isAzure is False")

        if isAzure:
            self.llm = AzureChatOpenAI(
                model=model,
                api_key=api_key,
                azure_endpoint=base_url,
                api_version="2024-02-01"
            )
        else:
            self.llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        
        self.text_parser = StrOutputParser()

    def get_llm(self):
        return self.llm

    def get_summary_template(self, template_type: str):
        if template_type == "leaf":
            return ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        """
                        You are a summarization assistant AI. 
                        Please summarize the text in Thai language on {topic} Topic.
                        Be concise and to the point.
                        Length must not exceed 500 characters.
                        """
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{input}"),
                ]
            )
        elif template_type == "cluster":
            return ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        """
                        You are a summarization assistant AI. 
                        Find the key points of the content in Thai language on {topic} Topic.
                        Be concise and to the point.
                        """
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{input}"),
                ]
            )
        elif template_type == "root":
            return ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        """
                        You are a summarization assistant AI. 
                        Summarize the following text in Thai language.
                        Follow the instructions below:
                        1. Read the content.
                        2. Summarize the text in Thai language using the JSON format:
                        - Attention Grabber
                        - Summary Part
                        - Instructor
                        - Target Audience
                        - Course Benefits
                        - Call to Action
                        - Hashtags
                        """
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{input}"),
                ]
            )

    def add_chat_history(self, chat_history: list, AIText: str, HumanText: str):
        chat_history.extend(
            [HumanMessage(content=HumanText), AIMessage(content=AIText)]
        )
        return chat_history

    def execute_chain(self, template: ChatPromptTemplate, input_data: dict, chat_history=[]) -> str:
        chain = template | self.llm | self.text_parser
        result = ""

        # Check if 'text' exists in the input_data to split into chunks
        if "text" in input_data:
            text_parser = TokenTextSplitter.from_defaults(chunk_overlap=0, chunk_size=4096)
            text_chunks = text_parser.split_text(text=input_data["text"])

            # Process each text chunk
            for chunk in text_chunks:
                input_data["input"] = chunk
                latest_chat_history = chat_history[-1:]
                res = chain.invoke(
                    {**input_data, "chat_history": latest_chat_history}
                )
                self.add_chat_history(chat_history, HumanText=chunk, AIText=res)
                result += res
        else:
            # If no 'text' to chunk, just pass the input_data as is
            latest_chat_history = chat_history[-1:]
            res = chain.invoke({**input_data, "chat_history": latest_chat_history})
            self.add_chat_history(chat_history, HumanText=input_data.get("input", ""), AIText=res)
            result = res

        return result

    def summarize_transcription(self, text: str, topic: str, template_type: str) -> str:
        summary_template = self.get_summary_template(template_type)
        chat_history = []
        input_data = {"topic": topic, "text": text}
        result = self.execute_chain(summary_template, input_data, chat_history)
        return result

    def summarize_root_transcription(self, text: str, instructor: str, link2course: str) -> str:
        summary_root_template = self.get_summary_template("root")
        chat_history = []

        # Prepare the input content
        input_content = f"""
            Content: {text}
            Instructor: {instructor}
            Link to course: {link2course}
        """

        # First execution of the chain for summarization
        input_data = {"input": input_content}  # input for the chain
        result = self.execute_chain(summary_root_template, input_data, chat_history)

        # Now that we have the first result (the summary), we want to format it as JSON.
        json_input = f"""
            From the following content, make it a JSON format. Don't MISS any text, emoji from the content. COPY IT!
            Think step by step and be concise.

            JSON format:
            {{
                "attentionGrabber": ,
                "summaryPart": ,
                "instructor": ,
                "targetAudience": ,
                "courseBenefit": ,
                "callToAction": ,
                "hashtag": ,
            }}
            Content: {result}  # Use the result from the previous step
        """

        # Second execution of the chain to convert the summary into the required JSON format
        json_input_data = {"input": json_input}  # input for the second chain invocation
        json_result = self.execute_chain(summary_root_template, json_input_data, chat_history)

        return json_result