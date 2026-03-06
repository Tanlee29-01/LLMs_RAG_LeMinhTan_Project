from typing import Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import torch

# Tự động nhận diện: Giờ máy bạn có CUDA, nó sẽ tự chọn "cuda"


class VectorDB:
    def __init__(self,
                documents = None,
                vector_db: Union[Chroma,FAISS] = Chroma,
                # Truyền biến device vào đây
                embedding = HuggingFaceEmbeddings(model_kwargs = {'device': 'cuda:0'})
                ) -> None: 

        self.vector_db = vector_db
        self.embedding = embedding
        self.db = self._build_db(documents)

    def _build_db(self,documents):
        self.db = self.vector_db.from_documents(documents=documents,embedding=self.embedding)
        return self.db

    def get_retriever(self,
                    search_type :str = "similarity",
                    search_kwargs: dict = {"k":10}
                    ):
        retriever = self.db.as_retriever(search_type = search_type,
                                        search_kwargs= search_kwargs)
        return retriever