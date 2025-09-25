from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(documents, embedding_model, 'my_colletction')

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query = 'What is chroma is used for?'
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(i, doc.page_content)