import glob
import multiprocessing
import os
import re
from typing import List, Union

import pymupdf4llm
from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from tqdm import tqdm


def remove_non_utf8_character(text: str) -> str:
    # Keep full Unicode content (including Vietnamese) and drop only control chars.
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)


def _get_markdown_splitter() -> MarkdownHeaderTextSplitter:
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
        strip_headers=False,
    )


def load_pdf(pdf_file: str):
    docs = PyPDFLoader(pdf_file, extract_images=True).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_character(doc.page_content)
    return docs


def load_pdf_structural(pdf_file: str):
    md_text = pymupdf4llm.to_markdown(pdf_file)
    md_header_splits = _get_markdown_splitter().split_text(md_text)

    for doc in md_header_splits:
        doc.page_content = remove_non_utf8_character(doc.page_content)
        doc.metadata = {**(doc.metadata or {}), "source": pdf_file, "loader": "pymupdf4llm"}

    return md_header_splits


def load_pdf_docling(pdf_file: str, converter: DocumentConverter):
    conversion_result = converter.convert(pdf_file)
    md_text = conversion_result.document.export_to_markdown()
    if not md_text or not md_text.strip():
        raise ValueError(f"Docling returned empty content for file: {pdf_file}")

    md_header_splits = _get_markdown_splitter().split_text(md_text)
    if not md_header_splits:
        md_header_splits = [Document(page_content=md_text, metadata={})]

    for doc in md_header_splits:
        doc.page_content = remove_non_utf8_character(doc.page_content)
        doc.metadata = {**(doc.metadata or {}), "source": pdf_file, "loader": "docling"}

    return md_header_splits


def get_num_cpu():
    return multiprocessing.cpu_count()


class BaseLoader:
    def __init__(self):
        self.num_process = get_num_cpu()

    def __call__(self, *args, **kwds):
        pass


class PDFLoader(BaseLoader):
    def __init__(self, strategy: str = "docling") -> None:
        super().__init__()
        self.strategy = strategy
        self.docling_converter = DocumentConverter() if strategy == "docling" else None

    def _load_single_file(self, pdf_file: str):
        if self.strategy == "docling":
            try:
                return load_pdf_docling(pdf_file, self.docling_converter)
            except Exception as exc:
                print(f"[WARN] Docling failed for {pdf_file}. Fallback to pymupdf4llm. Error: {exc}")
                return load_pdf_structural(pdf_file)

        if self.strategy == "structural":
            return load_pdf_structural(pdf_file)

        return load_pdf(pdf_file)

    def __call__(self, pdf_files: List[str], **kwargs):
        num_process = min(self.num_process, kwargs.get("workers", 4))

        # For Docling on Windows, run sequentially to avoid startup and permissions issues.
        if self.strategy == "docling" or num_process <= 1 or os.name == "nt":
            doc_loaded = []
            with tqdm(total=len(pdf_files), desc=f"Loading PDFs ({self.strategy})", unit="file") as pbar:
                for pdf_file in pdf_files:
                    doc_loaded.extend(self._load_single_file(pdf_file))
                    pbar.update(1)
            return doc_loaded

        target_func = load_pdf_structural if self.strategy == "structural" else load_pdf
        with multiprocessing.Pool(processes=num_process) as pool:
            doc_loaded = []
            with tqdm(total=len(pdf_files), desc=f"Loading PDFs ({self.strategy})", unit="file") as pbar:
                for res in pool.imap_unordered(target_func, pdf_files):
                    doc_loaded.extend(res)
                    pbar.update(1)

        return doc_loaded


class TextSplitter:
    def __init__(
        self,
        separators: List[str] = ["\n\n", "\n", " ", "", "."],
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def __call__(self, documents):
        return self.splitter.split_documents(documents)


class Loader:
    def __init__(
        self,
        file_type: str = "pdf",
        strategy: str = "docling",
        split_kwargs: dict = None,
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

        doc_loader = self.doc_loader(pdf_files, workers=workers)
        doc_split = self.doc_splitter(doc_loader)

        return doc_split

    def load_dir(self, dir_path: str, worker: int = 2):
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*pdf")
            assert len(files) > 0, f"No {self.file_type} file found in {dir_path}"
        else:
            raise ValueError("file_type must be pdf")
        return self.load(files, workers=worker)