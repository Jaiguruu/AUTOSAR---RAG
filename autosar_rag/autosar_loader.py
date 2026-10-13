from typing import AsyncIterator, Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pypdf import PdfReader
from io import BytesIO

class AutosarLoader(BaseLoader):

    def __init__(self, uploaded_file) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.uploaded_file = uploaded_file


    def lazy_load(self) -> Iterator[Document]:
        # A lazy loader that reads a file.
        stream = BytesIO(self.uploaded_file.read())
        with PdfReader(stream) as reader:
            for page_number, page in enumerate(reader.pages):
                str_list = []
                def visitor_body(text, cm, tm, font_dict, font_size):
                    nonlocal str_list
                    y = tm[5]
                    if 50 < y < 750:
                        if not text.endswith("\n") and not text.endswith(" "):
                            text += " "
                        text = text.removeprefix(" ")
                        str_list.append(text)

                page.extract_text(visitor_text=visitor_body)
                yield Document(
                    page_content="".join(str_list),
                    metadata={"page_number": page_number + 1, "source": self.uploaded_file.name},
                )


if __name__ == "__main__":

    from pathlib import Path
    cookbook_dir = Path(__file__).parent
    pdf_path = cookbook_dir.joinpath("data/AUTOSAR/AUTOSAR_CP_SWS_DiagnosticCommunicationManager-54-78.pdf")

    loader = AutosarLoader(str(pdf_path))

    ## Test out the lazy load interface
    for doc in loader.lazy_load():
        print(doc, '\n')
