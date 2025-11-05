from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
import dotenv

dotenv.load_dotenv()

from os import getenv

vector_store = Milvus(
    connection_args={'uri': getenv("MILVUS_URI")},
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# from pymilvus import MilvusClient
# client = MilvusClient(uri=getenv("MILVUS_URI"))

print(Document("I am a cat.", metadata={'source': "https://example.com"}))