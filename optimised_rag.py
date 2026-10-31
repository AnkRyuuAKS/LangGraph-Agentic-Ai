
import os
import json
import hashlib
import requests

from pathlib import Path
from dotenv import load_dotenv
from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
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

INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)


# ==================================================
# JINA EMBEDDINGS
# ==================================================

class JinaAPIEmbeddings(Embeddings):

    def __init__(
        self,
        model="jina-embeddings-v5-text-nano",
        dimensions=512,
        batch_size=32
    ):

        self.api_key = os.getenv("JINA_API_KEY")

        if not self.api_key:
            raise ValueError("JINA_API_KEY missing in .env")

        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size

        self.url = "https://api.jina.ai/v1/embeddings"

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


EMBED_MODEL_NAME = "jina-embeddings-v5-text-nano"
EMBED_DIMENSIONS = 512

emb = JinaAPIEmbeddings(
    model=EMBED_MODEL_NAME,
    dimensions=EMBED_DIMENSIONS,
    batch_size=32
)


# ==================================================
# PDF LOADING AND SPLITTING
# ==================================================

@traceable(name="load_pdf")
def load_pdf(path: str):

    return PyPDFLoader(path).load()


@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(docs)


# ==================================================
# BUILD VECTOR STORE
# ==================================================

@traceable(name="build_vectorstore")
def build_vectorstore(splits):

    if not splits:
        raise ValueError("No document chunks found.")

    return FAISS.from_documents(splits, emb)


# ==================================================
# CACHE KEY / FINGERPRINT
# ==================================================

def _file_fingerprint(path: str) -> dict:

    p = Path(path)

    h = hashlib.sha256()

    with p.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return {
        "sha256": h.hexdigest(),
        "size": p.stat().st_size,
        "mtime": int(p.stat().st_mtime)
    }


def _index_key(
    pdf_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_model_name: str
) -> str:

    meta = {

        "pdf_fingerprint": _file_fingerprint(pdf_path),

        "chunk_size": chunk_size,

        "chunk_overlap": chunk_overlap,

        "embedding_model": embed_model_name,

        "embedding_dimensions": EMBED_DIMENSIONS,

        "format": "v2"
    }

    return hashlib.sha256(
        json.dumps(meta, sort_keys=True).encode("utf-8")
    ).hexdigest()


# ==================================================
# LOAD EXISTING INDEX
# ==================================================

@traceable(name="load_index", tags=["index"])
def load_index_run(index_dir: Path):

    return FAISS.load_local(
        str(index_dir),
        emb,
        allow_dangerous_deserialization=True
    )


# ==================================================
# BUILD AND SAVE INDEX
# ==================================================

@traceable(name="build_index", tags=["index"])
def build_index_run(
    pdf_path: str,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    embed_model_name: str
):

    docs = load_pdf(pdf_path)

    splits = split_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    print("Total chunks:", len(splits))

    vs = build_vectorstore(splits)

    index_dir.mkdir(parents=True, exist_ok=True)

    vs.save_local(str(index_dir))

    (index_dir / "meta.json").write_text(
        json.dumps({
            "pdf_path": os.path.abspath(pdf_path),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embed_model_name,
            "embedding_dimensions": EMBED_DIMENSIONS
        }, indent=2)
    )

    print("Vector index saved successfully.")

    return vs


# ==================================================
# LOAD OR BUILD INDEX
# ==================================================

def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = EMBED_MODEL_NAME,
    force_rebuild: bool = False
):

    key = _index_key(
        pdf_path,
        chunk_size,
        chunk_overlap,
        embed_model_name
    )

    index_dir = INDEX_ROOT / key

    index_exists = (
        (index_dir / "index.faiss").is_file()
        and (index_dir / "index.pkl").is_file()
    )

    if index_exists and not force_rebuild:

        print("Cache HIT: Loading existing FAISS index.")

        return load_index_run(index_dir)

    print("Cache MISS: Building new FAISS index.")

    return build_index_run(
        pdf_path,
        index_dir,
        chunk_size,
        chunk_overlap,
        embed_model_name
    )


# ==================================================
# GROQ LLM
# ==================================================

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=800
)


# ==================================================
# PROMPT
# ==================================================

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

    return "\n\n".join(
        d.page_content for d in docs
    )


# ==================================================
# SETUP PIPELINE
# ==================================================

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(
    pdf_path: str,
    chunk_size=1000,
    chunk_overlap=150,
    embed_model_name=EMBED_MODEL_NAME,
    force_rebuild=False
):

    return load_or_build_index(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        force_rebuild=force_rebuild
    )


# ==================================================
# FULL RAG PIPELINE
# ==================================================

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = EMBED_MODEL_NAME,
    force_rebuild: bool = False
):

    vectorstore = setup_pipeline(
        pdf_path,
        chunk_size,
        chunk_overlap,
        embed_model_name,
        force_rebuild
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    parallel = RunnableParallel({

        "context": retriever | RunnableLambda(format_docs),

        "question": RunnablePassthrough(),

    })

    chain = parallel | prompt | llm | StrOutputParser()

    return chain.invoke(

        question,

        config={
            "run_name": "pdf_rag_query",
            "tags": ["qa"],
            "metadata": {"k": 4}
        }

    )


# ==================================================
# CLI
# ==================================================

if __name__ == "__main__":

    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")

    q = input("\nQ: ").strip()

    if not q:
        print("Please enter a question.")
    else:

        ans = setup_pipeline_and_query(
            PDF_PATH,
            q
        )

        print("\nA:", ans)
