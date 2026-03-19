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

def build_rag_chain(llm,data_dir,data_type):
    doc_loaded = Loader(file_type = data_type).load_dir(data_dir,worker=2)
    retriever = VectorDB(documents = doc_loaded).get_retriever()
    rag_train = Offline_RAG(llm).get_chain(retriever)

    return rag_train