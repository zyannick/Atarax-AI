import tiktoken
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ataraxai.app_logic.modules.rag.parser.document_base_parser import DocumentChunk
from langchain_core.documents import Document
from pathlib import Path
from ataraxai.app_logic.modules.rag.parser.pdf_parser import PDFParser
from ataraxai.app_logic.modules.rag.parser.docx_parser import DOCXParser
from ataraxai.app_logic.modules.rag.parser.pptx_parser import PPTXParser
from typing import Optional, Dict, Any, Callable
from typing_extensions import Union


EXT_PARSER_MAP: Dict[str, Any] = {
    ".pdf": PDFParser(),
    ".docx": DOCXParser(),
    ".pptx": PPTXParser(),
}


class SmartChunker:
    def __init__(
        self,
        model_name_for_tiktoken: str = "gpt-3.5-turbo",
        chunk_size_tokens: int = 400,
        chunk_overlap_tokens: int = 50,
        separators: List[str] | None = None,
        keep_separator: bool = True,
    ):
        try:
            self._tokenizer = tiktoken.encoding_for_model(model_name_for_tiktoken)
        except KeyError:
            print(
                f"Warning: Model '{model_name_for_tiktoken}' not found for tiktoken. Defaulting to 'cl100k_base' for token counting."
            )
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError(
                "Chunk overlap (in tokens) must be smaller than chunk size (in tokens)."
            )

        self._length_function: Callable[[str], int] = lambda text: len(
            self._tokenizer.encode(text, disallowed_special=())
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
            length_function=self._length_function,
            separators=separators,
            keep_separator=keep_separator,
            add_start_index=True,
        )
        print(
            f"SmartChunker initialized with LangChain RecursiveCharacterTextSplitter: "
            f"chunk_size={chunk_size_tokens} tokens, overlap={chunk_overlap_tokens} tokens."
        )

    def ingest_file(self, file_path: Path) -> List[DocumentChunk]:
        """
        Ingests a single file and returns a list of DocumentChunk objects.
        """
        if not file_path.is_file():
            print(f"[!] {file_path} is not a valid file.")
            return []

        parser = EXT_PARSER_MAP.get(file_path.suffix.lower())
        if not parser:
            print(f"[!] No parser available for {file_path.suffix}. Skipping.")
            return []

        print(f"[+] Parsing: {file_path}")
        try:
            raw_chunks: List[DocumentChunk] = parser.parse(file_path)
            smart_chunks = self.chunk(raw_chunks)
            return smart_chunks
        except Exception as e:
            print(f"[!] Failed to process {file_path}: {e}")
            return []

    def ingest_directory(self, directory: Path) -> List[DocumentChunk]:
        all_chunks: List[DocumentChunk] = []

        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            parser = EXT_PARSER_MAP.get(path.suffix.lower())
            if parser:
                print(f"[+] Parsing: {path}")
                try:
                    raw_chunks: List[DocumentChunk] = parser.parse(path)
                    smart_chunks = self.chunk(raw_chunks)
                    all_chunks.extend(smart_chunks)
                except Exception as e:
                    print(f"[!] Failed to process {path}: {e}")
        return all_chunks

    def _chunk_single_document_content(
        self, document_content: str, source_path: str, base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunks a single document's content string using the initialized LangChain text splitter.
        """
        if not document_content or not document_content.strip():
            return []

        langchain_doc_metadata: Dict[str, Any] = {
            "original_source": source_path,
            **base_metadata,
        }

        langchain_documents: List[Document] = self.text_splitter.create_documents(  # type: ignore
            texts=[document_content], metadatas=[langchain_doc_metadata]
        )

        final_chunks: List[DocumentChunk] = []
        for i, lc_doc in enumerate(langchain_documents):
            chunk_specific_metadata: Dict[str, Any] = lc_doc.metadata.copy()  # type: ignore
            chunk_specific_metadata["chunk_index_in_doc"] = i

            final_chunks.append(
                DocumentChunk(
                    content=lc_doc.page_content,
                    source=source_path,
                    metadata=chunk_specific_metadata,
                )
            )

        print(
            f"SmartChunker: Document from '{source_path}' split into {len(final_chunks)} chunks."
        )
        return final_chunks

    def chunk(self, documents_to_chunk: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Takes a list of DocumentChunk objects (where each 'content' field is a full document text)
        and returns a new list of DocumentChunk objects where content has been chunked.
        """
        all_resulting_chunks: List[DocumentChunk] = []
        for original_doc_chunk in documents_to_chunk:
            if not isinstance(original_doc_chunk, DocumentChunk):
                print(
                    f"Warning: SmartChunker.chunk expected DocumentChunk, got {type(original_doc_chunk)}. Skipping."
                )
                continue

            newly_chunked_parts = self._chunk_single_document_content(
                document_content=original_doc_chunk.content,
                source_path=original_doc_chunk.source,
                base_metadata=original_doc_chunk.metadata,
            )
            all_resulting_chunks.extend(newly_chunked_parts)
        return all_resulting_chunks
