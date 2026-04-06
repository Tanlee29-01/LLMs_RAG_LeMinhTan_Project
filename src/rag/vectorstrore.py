from typing import Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
import os
from langchain_ollama import OllamaEmbeddings

DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")


class VectorDB:
    def __init__(self,
                documents = None,
                vector_db: Union[Chroma,FAISS] = Chroma,
                embedding = None,
                metadata_filter: dict | None = None,
                ) -> None:

        self.documents = documents or []
        self.vector_db = vector_db
        self.embedding = embedding or OllamaEmbeddings(
            model=DEFAULT_EMBEDDING_MODEL,
            base_url=DEFAULT_OLLAMA_BASE_URL,
        )
        self.metadata_filter = metadata_filter or {}
        self.db = self._build_db(documents)

    def _build_db(self,documents):
        self.db = self.vector_db.from_documents(documents=documents,embedding=self.embedding)
        return self.db

    def get_retriever(self,
                    search_type: str = "mmr",
                    search_kwargs: dict = None,
                    ):
        if search_kwargs is None:
            search_kwargs = {"k": 4, "fetch_k": 12}
        if self.metadata_filter:
            search_kwargs = {**search_kwargs, "filter": self.metadata_filter}
        retriever = self.db.as_retriever(search_type=search_type,
                                        search_kwargs=search_kwargs)
        return retriever

    def _sorted_documents(self):
        def sort_key(doc):
            meta = doc.metadata or {}
            source = meta.get("source", "")
            page = meta.get("page")
            page = page if page is not None else 10**9
            header_1 = meta.get("Header 1", "")
            header_2 = meta.get("Header 2", "")
            return (source, page, header_1, header_2)

        return sorted(self.documents, key=sort_key)

    def get_summary_documents(self, max_docs: int = 12, max_chars: int = 12000):
        ordered_docs = self._sorted_documents()
        if not ordered_docs:
            return []

        sample_size = min(max_docs, len(ordered_docs))
        if sample_size == len(ordered_docs):
            candidate_indices = list(range(len(ordered_docs)))
        elif sample_size == 1:
            candidate_indices = [0]
        else:
            candidate_indices = []
            for index in range(sample_size):
                mapped_index = round(index * (len(ordered_docs) - 1) / (sample_size - 1))
                if mapped_index not in candidate_indices:
                    candidate_indices.append(mapped_index)

        summary_docs = []
        total_chars = 0
        for index in candidate_indices:
            doc = ordered_docs[index]
            doc_length = len(doc.page_content or "")
            if summary_docs and total_chars + doc_length > max_chars:
                break
            summary_docs.append(doc)
            total_chars += doc_length

        return summary_docs or ordered_docs[:1]