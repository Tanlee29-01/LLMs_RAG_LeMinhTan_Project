import os
import re
from threading import RLock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from models.llm_model import get_hf_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA

llm = get_hf_llm(temperature=0.1, model_name="llama3.1:8b")
genai_docs = "data/generative_ai"
default_chat_id = "default"

chat_chain_cache = {}
chat_chain_lock = RLock()


def normalize_chat_id(chat_id: str | None) -> str:
    if not chat_id:
        return default_chat_id

    normalized = re.sub(r"[^a-zA-Z0-9._-]", "_", str(chat_id).strip())
    return normalized or default_chat_id


def get_chat_dir(chat_id: str | None) -> str:
    normalized_chat_id = normalize_chat_id(chat_id)
    return os.path.join(genai_docs, normalized_chat_id)


def build_chat_chain(chat_id: str | None):
    normalized_chat_id = normalize_chat_id(chat_id)
    return build_rag_chain(
        llm,
        data_dir=genai_docs,
        data_type="mixed",
        chat_id=normalized_chat_id,
    )


def get_chat_chain(chat_id: str | None):
    normalized_chat_id = normalize_chat_id(chat_id)
    with chat_chain_lock:
        chain = chat_chain_cache.get(normalized_chat_id)
        if chain is None:
            chain = build_chat_chain(normalized_chat_id)
            chat_chain_cache[normalized_chat_id] = chain
        return chain


def rebuild_chain(chat_id: str | None = None):
    """Rebuild RAG chain with latest documents for one chat."""
    normalized_chat_id = normalize_chat_id(chat_id)
    with chat_chain_lock:
        chat_chain_cache.pop(normalized_chat_id, None)

    try:
        chain = build_chat_chain(normalized_chat_id)
        with chat_chain_lock:
            chat_chain_cache[normalized_chat_id] = chain
        return {"status": "success", "message": "Chain rebuilt with latest documents", "chat_id": normalized_chat_id}
    except Exception as e:
        return {"status": "error", "message": str(e), "chat_id": normalized_chat_id}


#==============chain==============

genai_chain = get_chat_chain(default_chat_id)


# uvicorn src.app:app --host "0.0.0.0" --port 5051 --reload
#-> click here http://localhost:5051/docs 

#==============App - FastAPI==============
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
    expose_headers = ["*"]
)


#==============Routes - FastAPi==============

@app.get("/")
async def root():
    return {
        "message": "RAG API is running",
        "health": "/check",
        "docs": "/docs",
        "inference": "/generative_ai",
        "upload": "/upload",
    }

@app.get("/check")
async def check():
    return {"status":"ok"}

@app.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    chat_id: str | None = Form(default=None),
):
    """Nhận và lưu tài liệu tải lên, sau đó rebuild chain"""
    try:
        normalized_chat_id = normalize_chat_id(chat_id)
        chat_dir = get_chat_dir(normalized_chat_id)
        os.makedirs(chat_dir, exist_ok=True)
        
        saved_files = []
        for file in files:
            safe_filename = os.path.basename(file.filename or "")
            if not safe_filename.lower().endswith((".pdf", ".txt")):
                raise HTTPException(status_code=400, detail=f"Only PDF/TXT accepted: {safe_filename}")
            
            file_path = os.path.join(chat_dir, safe_filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(safe_filename)
        
        # Rebuild chain với tài liệu mới
        result = rebuild_chain(normalized_chat_id)
        return {
            "status": "success",
            "uploaded_files": saved_files,
            "chat_id": normalized_chat_id,
            "storage_dir": chat_dir,
            "chain_status": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generative_ai",response_model=OutputQA)
async def generative_ai(inputs : InputQA):
    chat_id = normalize_chat_id(getattr(inputs, "chat_id", None))
    try:
        chain = get_chat_chain(chat_id)
        result = chain.invoke(inputs.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG inference failed: {exc}") from exc

    if isinstance(result, dict):
        answer = result.get("answer")
        source = result.get("source")
    else:
        answer = result
        source = None

    if hasattr(answer, "content"):
        answer = answer.content

    if answer is None:
        raise HTTPException(status_code=500, detail="RAG inference returned empty answer")

    return {"answer": str(answer), "source": source, "chat_id": chat_id}

#==============Langserver Routes - Playground==============
add_routes(
    app,
    genai_chain,
    playground_type="default",
    path="/langserve/generative_ai"
)