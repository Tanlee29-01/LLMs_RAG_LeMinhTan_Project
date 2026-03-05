import glob
import multiprocessing
from tqdm import tqdm
from typing import Union,List,Literal
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def remove_non_utf8_character(text):
    return ''.join(char for char in text if ord(char) < 128)
def load_pdf(pdf_file):
    docs = PyPDFLoader(pdf_file,extract_images=True).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_character(doc.page_content)

    return docs
def get_num_cpu():
    return multiprocessing.cpu_count()

class BaseLoader:
    def __init__(self):
        self.num_process = get_num_cpu()

    def __call__(self, *args, **kwds):
        pass

class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pdf_files:List[str], **kwargs):
        num_process = min(self.num_process,kwargs["workers"])
        with multiprocessing.Pool(processes=num_process) as pool:
            doc_loaded = []
            total_files = len(pdf_files)
            with tqdm(total=total_files,desc="Loading PDFs",unit="file") as pbar:
                for res in pool.imap_unordered(load_pdf,pdf_files):
                    doc_loaded.extend(res)
                    pbar.update(1)

        return doc_loaded
    

class TextSplitter:
    def __init__(self,
                separators: List[str] = ['\n\n','\n',' ',''],
                chunk_size: int = 300,
                chunk_overlap:int = 0
                ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )

    def __call__(self,documents):
        return self.splitter.split_documents(documents)


class Loader:
    def __init__(self,
                file_type:str = Literal["pdf"],
                split_kwargs: dict = {
                    "chunk_size":300,
                    "chunk_overlap":0,
                }
                ) -> None:
        assert file_type in ["pdf"], "file_type must be pdf"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        else:
            raise ValueError("file_type must be pdf")
        
        self.doc_splitter = TextSplitter(**split_kwargs)

    def load(self,pdf_files:Union[str,List[str]],workers:int = 1):
        if isinstance(pdf_files,str):
            pdf_files = [pdf_files]
        doc_loader = self.doc_loader(pdf_files,workers= workers)
        doc_split = self.doc_splitter(doc_loader)
        return doc_split

    def load_dir(self,dir_path:str,worker:int = 2):
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*pdf")
            assert len(files) > 0, f"No {self.file_type} file found in {dir_path}"
        else:
            raise ValueError("file_type must be pdf")
        return self.load(files,workers=worker)