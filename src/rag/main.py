import os

from pydantic import AliasChoices, BaseModel, Field

from src.rag.file_loader import Loader
from src.rag.vectorstrore import VectorDB
from src.rag.offline_rag import Offline_RAG


class InputQA(BaseModel):
    question: str = Field(
        ...,
        title="Question to ask the model",
        validation_alias=AliasChoices("question", "quesion"),
        serialization_alias="question",
    )
    chat_id: str | None = Field(default=None, title="Chat identifier")

class OutputQA(BaseModel):
    answer:str = Field(...,title="Answer from model")
    source: str | None = Field(default=None, title="Answer source")
    chat_id: str | None = Field(default=None, title="Chat identifier")

def _resolve_chat_dir(data_dir: str, chat_id: str | None):
    if not chat_id:
        return data_dir
    return os.path.join(data_dir, chat_id)


def build_rag_chain(llm, data_dir, data_type, mode: str = "qa", chat_id: str | None = None):
    rag = Offline_RAG(llm)
    target_dir = _resolve_chat_dir(data_dir, chat_id)
    metadata = {"user_id": chat_id} if chat_id else None

    try:
        doc_loaded = Loader(file_type=data_type, strategy="docling").load_dir(
            target_dir,
            worker=1,
            metadata=metadata,
        )
    except AssertionError:
        return rag.get_general_chain()

    if not doc_loaded:
        return rag.get_general_chain()

    retriever = VectorDB(documents=doc_loaded, metadata_filter=metadata).get_retriever()
    rag_train = rag.get_chain(retriever, mode=mode)

    return rag_train
