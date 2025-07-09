import tiktoken
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ataraxai.app_logic.modules.rag.parser.document_base_parser import DocumentChunk
from langchain_core.documents import Document
from pathlib import Path
from ataraxai.app_logic.modules.rag.parser.pdf_parser import PDFParser
from ataraxai.app_logic.modules.rag.parser.docx_parser import DOCXParser
from ataraxai.app_logic.modules.rag.parser.pptx_parser import PPTXParser
from typing import Callable


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
        """
        Initializes the SmartChunker with a tokenizer and a recursive character text splitter.

        Args:
            model_name_for_tiktoken (str): Name of the model to use for tiktoken encoding. Defaults to "gpt-3.5-turbo".
            chunk_size_tokens (int): The maximum number of tokens per chunk. Defaults to 400.
            chunk_overlap_tokens (int): The number of tokens to overlap between chunks. Must be less than chunk_size_tokens. Defaults to 50.
            separators (List[str] | None): Optional list of separator strings to use when splitting text. Defaults to None.
            keep_separator (bool): Whether to keep the separator at the end of each chunk. Defaults to True.

        Raises:
            ValueError: If chunk_overlap_tokens is greater than or equal to chunk_size_tokens.

        Notes:
            - If the specified model is not found in tiktoken, defaults to "cl100k_base" encoding.
            - Uses LangChain's RecursiveCharacterTextSplitter for chunking text.
        """
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
        Ingests a file, parses its content into document chunks, and applies smart chunking.

        Args:
            file_path (Path): The path to the file to be ingested.

        Returns:
            List[DocumentChunk]: A list of smart-chunked DocumentChunk objects extracted from the file.
                                 Returns an empty list if the file is invalid, unsupported, or an error occurs.

        Side Effects:
            Prints status and error messages to the console.
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
        """
        Ingests all files from the specified directory, recursively parsing and chunking their contents.

        Args:
            directory (Path): The root directory to search for files.

        Returns:
            List[DocumentChunk]: A list of processed and chunked document objects extracted from all files.

        Notes:
            - Only files with extensions present in EXT_PARSER_MAP are processed.
            - Each file is parsed using its corresponding parser, then further chunked using the `chunk` method.
            - Files that cannot be processed are skipped, and an error message is printed.
        """
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
        Splits the content of a single document into smaller chunks using the configured text splitter,
        and attaches relevant metadata to each chunk.

        Args:
            document_content (str): The full text content of the document to be chunked.
            source_path (str): The original source path or identifier of the document.
            base_metadata (Dict[str, Any]): Base metadata to be included with each chunk.

        Returns:
            List[DocumentChunk]: A list of DocumentChunk objects, each containing a chunk of the original
            document content along with associated metadata, including the chunk's index within the document.

        Notes:
            - If the document content is empty or contains only whitespace, an empty list is returned.
            - Each chunk's metadata includes the original source, base metadata, and its index in the document.
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
        Chunks a list of DocumentChunk objects using the smart chunking strategy.

        Args:
            documents_to_chunk (List[DocumentChunk]): A list of DocumentChunk instances to be chunked.

        Returns:
            List[DocumentChunk]: A list containing the resulting DocumentChunk objects after chunking.

        Notes:
            - If an item in documents_to_chunk is not an instance of DocumentChunk, it will be skipped with a warning.
            - Each DocumentChunk is processed by the _chunk_single_document_content method, which may split it into multiple chunks.
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
