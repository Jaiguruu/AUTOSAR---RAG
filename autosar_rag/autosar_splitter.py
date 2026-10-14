from collections.abc import Iterable
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class AutosarSplitter:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 30):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                ". ",
                ", ",
                " ",
                "",
            ],
        )

    def split_documents(
        self, documents: Iterable[Document]
    ) -> List[Document]:
        return self.text_splitter.split_documents(documents)
