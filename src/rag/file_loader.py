import glob
import multiprocessing
import os
import re
from tqdm import tqdm
from typing import Union, List, Literal
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import pymupdf4llm  

def remove_non_utf8_character(text):
    # Keep full Unicode content (including Vietnamese) and drop only control chars.
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

# --- CÁCH 1: LOAD THEO TRANG PDF CƠ BẢN ---
def load_pdf(pdf_file):
    docs = PyPDFLoader(pdf_file, extract_images=True).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_character(doc.page_content)
    return docs

# --- CÁCH 2: LOAD VÀ CẮT THEO CẤU TRÚC MARKDOWN (THÊM MỚI) ---
def load_pdf_structural(pdf_file):
    """
    Đọc PDF, chuyển thành Markdown và cắt theo thẻ Header.
    Hàm này chạy độc lập cho từng file trong multiprocessing.
    """
    # 1. Chuyển PDF sang Markdown
    md_text = pymupdf4llm.to_markdown(pdf_file)
    
    # 2. Định nghĩa cấu trúc Header để cắt
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    # 3. Tiến hành cắt theo Header
    md_header_splits = markdown_splitter.split_text(md_text)
    
    # Làm sạch ký tự lạ
    for doc in md_header_splits:
        doc.page_content = remove_non_utf8_character(doc.page_content)
        
    return md_header_splits

def get_num_cpu():
    return multiprocessing.cpu_count()

class BaseLoader:
    def __init__(self):
        self.num_process = get_num_cpu()

    def __call__(self, *args, **kwds):
        pass

class PDFLoader(BaseLoader):
    def __init__(self, strategy: str = "standard") -> None:
        super().__init__()
        self.strategy = strategy

    def __call__(self, pdf_files: List[str], **kwargs):
        num_process = min(self.num_process, kwargs.get("workers", 4))
        
        # Quyết định dùng hàm load nào dựa trên chiến lược
        target_func = load_pdf_structural if self.strategy == "structural" else load_pdf

        if num_process <= 1 or os.name == "nt":
            doc_loaded = []
            total_files = len(pdf_files)
            with tqdm(total=total_files, desc=f"Loading PDFs ({self.strategy})", unit="file") as pbar:
                for pdf_file in pdf_files:
                    doc_loaded.extend(target_func(pdf_file))
                    pbar.update(1)
            return doc_loaded
        
        with multiprocessing.Pool(processes=num_process) as pool:
            doc_loaded = []
            total_files = len(pdf_files)
            # Cập nhật thanh tiến trình để hiển thị loại chiến lược đang dùng
            with tqdm(total=total_files, desc=f"Loading PDFs ({self.strategy})", unit="file") as pbar:
                for res in pool.imap_unordered(target_func, pdf_files):
                    doc_loaded.extend(res)
                    pbar.update(1)

        return doc_loaded
    

class TextSplitter:
    def __init__(self,
                separators: List[str] = ['\n\n', '\n', ' ', '','.'],
                chunk_size: int = 1200,    # [CẬP NHẬT] Tăng kích thước mặc định lên 1200
                chunk_overlap: int = 150   # [CẬP NHẬT] Thêm overlap mặc định là 150
                ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def __call__(self, documents):
        return self.splitter.split_documents(documents)


class Loader:
    def __init__(self,
                file_type: str = "pdf",
                strategy: str = "standard",  # [THÊM MỚI] Cấu hình chiến lược mặc định: "standard" hoặc "structural"
                split_kwargs: dict = None
                ) -> None:
        assert file_type in ["pdf"], "file_type must be pdf"
        self.file_type = file_type
        self.strategy = strategy
        
        if split_kwargs is None:
            split_kwargs = {
                "chunk_size": 1200,
                "chunk_overlap": 150,
            }
            
        if file_type == "pdf":
            self.doc_loader = PDFLoader(strategy=strategy)
        else:
            raise ValueError("file_type must be pdf")
        
        self.doc_splitter = TextSplitter(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
            
        # Bước 1: Load file (Nếu là structural, nó sẽ cắt theo Header ngay tại đây)
        doc_loader = self.doc_loader(pdf_files, workers=workers)
        
        # Bước 2: Đi qua Text Splitter
        # Mục đích: Đảm bảo không có mục nào (dù đã cắt theo Header) bị vượt quá chunk_size
        doc_split = self.doc_splitter(doc_loader)
        
        return doc_split

    def load_dir(self, dir_path: str, worker: int = 2):
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*pdf")
            assert len(files) > 0, f"No {self.file_type} file found in {dir_path}"
        else:
            raise ValueError("file_type must be pdf")
        return self.load(files, workers=worker)