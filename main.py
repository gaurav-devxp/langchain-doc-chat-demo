from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=100
)

loader = PyPDFLoader("documents/lucknow.pdf")
document = loader.load_and_split(
    text_splitter=text_splitter
)

vector_db = Chroma.from_documents(
    document,
    embedding=embeddings,
    persist_directory="emdb"
)

results = vector_db.similarity_search("How is Lucknow related to Brisbane", k=1)
# results_with_score = vector_db.similarity_search_with_score("How is Lucknow related to Brisbane", k=1)

# for result in results_with_score:
#     print("\n")
#     print(result[1])
#     print(result[0].page_content)

for result in results:
    print("\n")
    print(result.page_content)