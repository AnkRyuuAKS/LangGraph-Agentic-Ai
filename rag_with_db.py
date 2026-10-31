# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv langsmith


import os
import requests

from dotenv import load_dotenv
from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PDF_PATH = "cv_detail.pdf"


# ---------- Jina Embeddings ----------

class JinaAPIEmbeddings(Embeddings):

    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")
        self.model = "jina-embeddings-v5-text-nano"
        self.dimensions = 512
        self.batch_size = 32

        self.url = "https://api.jina.ai/v1/embeddings"

        if not self.api_key:
            raise ValueError("JINA_API_KEY not found in .env")

    def _embed(self, texts, task):

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):

            batch = texts[i:i + self.batch_size]

            response = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": batch,
                    "task": task,
                    "dimensions": self.dimensions,
                    "normalized": True
                },
                timeout=60
            )

            response.raise_for_status()

            data = response.json()["data"]

            data.sort(key=lambda x: x["index"])

            all_embeddings.extend(
                item["embedding"] for item in data
            )

        return all_embeddings

    def embed_documents(self, texts):
        return self._embed(texts, "retrieval.passage")

    def embed_query(self, text):
        return self._embed([text], "retrieval.query")[0]


emb = JinaAPIEmbeddings()


# ---------- traced setup steps ----------

@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()


@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=100):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(docs)


@traceable(name="build_vectorstore")
def build_vectorstore(splits):

    # FAISS internally calls Jina's API through emb
    vs = FAISS.from_documents(splits, emb)

    return vs


@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):

    docs = load_pdf(pdf_path)

    splits = split_documents(docs)

    vs = build_vectorstore(splits)

    return vs


# ---------- LLM ----------

llm = ChatGroq(
    model_name="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=800
)


# ---------- RAG Prompt ----------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY from the provided context. "
        "If not found, say you don't know."
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    )
])


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# ---------- Build Vector Store ----------

vectorstore = setup_pipeline(PDF_PATH)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
# ---------- RAG Chain ----------

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()


# ---------- Run Query ----------

print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")

q = input("\nQ: ").strip()

config = {
    "run_name": "pdf_rag_query"
}

ans = chain.invoke(q, config=config)

print("\nA:", ans)
