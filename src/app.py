import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from models.llm_model import get_hf_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA

llm = get_hf_llm(temperature=0.1, model_name="llama3.1:8b")
genai_docs = "data/generative_ai"


#==============chain==============

genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")


#==============Helper to rebuild chain==============
def rebuild_chain():
    """Rebuild RAG chain with latest documents"""
    global genai_chain
    try:
        genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")
        return {"status": "success", "message": "Chain rebuilt with latest documents"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


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
async def upload_documents(files: list[UploadFile] = File(...)):
    """Nhận và lưu tài liệu tải lên, sau đó rebuild chain"""
    try:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(genai_docs, exist_ok=True)
        
        saved_files = []
        for file in files:
            safe_filename = os.path.basename(file.filename or "")
            if not safe_filename.lower().endswith((".pdf", ".txt")):
                raise HTTPException(status_code=400, detail=f"Only PDF/TXT accepted: {safe_filename}")
            
            file_path = os.path.join(genai_docs, safe_filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(safe_filename)
        
        # Rebuild chain với tài liệu mới
        result = rebuild_chain()
        return {
            "status": "success",
            "uploaded_files": saved_files,
            "chain_status": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generative_ai",response_model=OutputQA)
async def generative_ai(inputs : InputQA):
    try:
        result = genai_chain.invoke(inputs.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG inference failed: {exc}") from exc

    if isinstance(result, dict):
        answer = result.get("answer")
    else:
        answer = result

    if hasattr(answer, "content"):
        answer = answer.content

    if answer is None:
        raise HTTPException(status_code=500, detail="RAG inference returned empty answer")

    return {"answer": str(answer)}

#==============Langserver Routes - Playground==============
add_routes(
    app,
    genai_chain,
    playground_type="default",
    path="/langserve/generative_ai"
)