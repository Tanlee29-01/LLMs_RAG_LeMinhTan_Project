import glob
import multiprocessing
import os
import re
from typing import Dict, List, Union

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


def _merge_metadata(documents, metadata: Dict[str, str] | None):
    if not metadata:
        return documents

    for doc in documents:
        doc.metadata = {**(doc.metadata or {}), **metadata}
    return documents


def load_pdf(pdf_file: str, metadata: Dict[str, str] | None = None):
    docs = PyPDFLoader(pdf_file, extract_images=True).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_character(doc.page_content)
    return _merge_metadata(docs, metadata)


def load_pdf_structural(pdf_file: str, metadata: Dict[str, str] | None = None):
    md_text = pymupdf4llm.to_markdown(pdf_file)
    md_header_splits = _get_markdown_splitter().split_text(md_text)

    for doc in md_header_splits:
        doc.page_content = remove_non_utf8_character(doc.page_content)
        doc.metadata = {**(doc.metadata or {}), "source": pdf_file, "loader": "pymupdf4llm"}

    return _merge_metadata(md_header_splits, metadata)


def load_pdf_docling(pdf_file: str, converter: DocumentConverter, metadata: Dict[str, str] | None = None):
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

    return _merge_metadata(md_header_splits, metadata)


def load_txt(text_file: str, metadata: Dict[str, str] | None = None):
    with open(text_file, "r", encoding="utf-8", errors="ignore") as file_handle:
        text = file_handle.read()

    document = Document(
        page_content=remove_non_utf8_character(text),
        metadata={"source": text_file, "loader": "text"},
    )
    return _merge_metadata([document], metadata)


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

    def _load_single_file(self, pdf_file: str, metadata: Dict[str, str] | None = None):
        if self.strategy == "docling":
            try:
                return load_pdf_docling(pdf_file, self.docling_converter, metadata=metadata)
            except Exception as exc:
                print(f"[WARN] Docling failed for {pdf_file}. Fallback to pymupdf4llm. Error: {exc}")
                return load_pdf_structural(pdf_file, metadata=metadata)

        if self.strategy == "structural":
            return load_pdf_structural(pdf_file, metadata=metadata)

        return load_pdf(pdf_file, metadata=metadata)

    def __call__(self, pdf_files: List[str], **kwargs):
        num_process = min(self.num_process, kwargs.get("workers", 4))
        metadata = kwargs.get("metadata")

        # For Docling on Windows, run sequentially to avoid startup and permissions issues.
        if self.strategy == "docling" or num_process <= 1 or os.name == "nt":
            doc_loaded = []
            with tqdm(total=len(pdf_files), desc=f"Loading PDFs ({self.strategy})", unit="file") as pbar:
                for pdf_file in pdf_files:
                    doc_loaded.extend(self._load_single_file(pdf_file, metadata=metadata))
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
        assert file_type in ["pdf", "txt", "mixed", "auto"], "file_type must be pdf, txt, mixed or auto"
        self.file_type = file_type
        self.strategy = strategy

        if split_kwargs is None:
            split_kwargs = {
                "chunk_size": 1200,
                "chunk_overlap": 150,
            }

        if file_type in ["pdf", "mixed", "auto"]:
            self.doc_loader = PDFLoader(strategy=strategy)
        else:
            self.doc_loader = None

        self.doc_splitter = TextSplitter(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], workers: int = 1, metadata: Dict[str, str] | None = None):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]

        doc_loader = []
        pdf_targets = [file_path for file_path in pdf_files if file_path.lower().endswith(".pdf")]
        txt_targets = [file_path for file_path in pdf_files if file_path.lower().endswith(".txt")]

        if pdf_targets and self.doc_loader is not None:
            doc_loader.extend(self.doc_loader(pdf_targets, workers=workers, metadata=metadata))

        for text_file in txt_targets:
            doc_loader.extend(load_txt(text_file, metadata=metadata))

        doc_split = self.doc_splitter(doc_loader)
        doc_split = _merge_metadata(doc_split, metadata)

        return doc_split

    def load_dir(self, dir_path: Union[str, List[str]], worker: int = 2, metadata: Dict[str, str] | None = None):
        if isinstance(dir_path, str):
            dir_paths = [dir_path]
        else:
            dir_paths = dir_path

        files = []
        for directory in dir_paths:
            if self.file_type in ["pdf", "mixed", "auto"]:
                files.extend(glob.glob(os.path.join(directory, "*.pdf")))
            if self.file_type in ["txt", "mixed", "auto"]:
                files.extend(glob.glob(os.path.join(directory, "*.txt")))

        files = list(dict.fromkeys(files))
        assert len(files) > 0, f"No {self.file_type} file found in {dir_paths}"
        return self.load(files, workers=worker, metadata=metadata)