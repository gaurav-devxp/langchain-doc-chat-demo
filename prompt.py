from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="How is Lucknow related to Brisbane")
args = parser.parse_args()

chat = ChatOpenAI()

memory = ConversationBufferMemory(
    memory_key="messages",
    return_messages=True
)

embeddings = OpenAIEmbeddings()
vector_db = Chroma(
    persist_directory="emdb",
    embedding_function=embeddings
)

retriever = vector_db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
    memory=memory
)

result = chain.run(args.prompt)

print(result)