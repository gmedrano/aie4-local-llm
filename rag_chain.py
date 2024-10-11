from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


url = "http://localhost:6333"
collection_name = "first_collection"

llm = OllamaLLM(
    model="llama3.2",
)

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
)

qdrant_vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings, collection_name=collection_name, url=url
)
retriever = qdrant_vectorstore.as_retriever()

RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
rag_chain = {"context": retriever | format_docs, "query": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()
