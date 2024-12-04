from openai import AzureOpenAI

class OpenAIEmbedding:
    def __init__(self, host, key) -> None:
        self.host = host
        self.key = key
        self.client = AzureOpenAI(
            azure_endpoint=self.host,
            api_key=self.key,
            api_version="2024-02-01",
        )

    def get_embedding(self, text: str):
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
            dimensions=1024,
        )

        return response.data[0].embedding