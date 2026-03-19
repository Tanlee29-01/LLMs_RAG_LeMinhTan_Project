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

class OutputQA(BaseModel):
    answer:str = Field(...,title="Answer from model")

def build_rag_chain(llm, data_dir, data_type, mode: str = "qa"):
    rag = Offline_RAG(llm)

    try:
        doc_loaded = Loader(file_type=data_type, strategy="docling").load_dir(data_dir, worker=1)
    except AssertionError:
        return rag.get_general_chain()

    if not doc_loaded:
        return rag.get_general_chain()

    retriever = VectorDB(documents=doc_loaded).get_retriever()
    rag_train = rag.get_chain(retriever, mode=mode)

    return rag_train
