import tiktoken
from typing import List, Dict, Any, Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ataraxai.app_logic.modules.rag.parser.document_base_parser import DocumentChunk


class SmartChunker:
    def __init__(self,
                 model_name_for_tiktoken: str = "gpt-3.5-turbo", 
                 chunk_size_tokens: int = 400,                   
                 chunk_overlap_tokens: int = 50,                 
                 separators: List[str] | None = None,          
                 keep_separator: bool = True                   
                 ):
        try:
            self._tokenizer = tiktoken.encoding_for_model(model_name_for_tiktoken)
        except KeyError:
            print(f"Warning: Model '{model_name_for_tiktoken}' not found for tiktoken. Defaulting to 'cl100k_base' for token counting.")
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("Chunk overlap (in tokens) must be smaller than chunk size (in tokens).")

        self._length_function = lambda text: len(self._tokenizer.encode(text, disallowed_special=()))

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
            length_function=self._length_function,
            separators=separators,  
            keep_separator=keep_separator,
            add_start_index=True   
        )
        print(f"SmartChunker initialized with LangChain RecursiveCharacterTextSplitter: "
              f"chunk_size={chunk_size_tokens} tokens, overlap={chunk_overlap_tokens} tokens.")

    def _chunk_single_document_content(self, document_content: str, source_path: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Chunks a single document's content string using the initialized LangChain text splitter.
        """
        if not document_content or not document_content.strip():
            return []

        # LangChain's create_documents expects a list of texts and a list of metadatas (one per text)
        # Here, we are chunking one large text, so we pass it as a single item in a list.
        # The metadata provided will be associated with the LangChain Document object
        # created from document_content, and then inherited by the chunks.
        
        # Prepare metadata for LangChain's Document object.
        # This metadata will be merged into each chunk's metadata by LangChain.
        langchain_doc_metadata = {"original_source": source_path, **base_metadata}
        
        langchain_documents = self.text_splitter.create_documents(
            texts=[document_content],       # Must be a list of texts
            metadatas=[langchain_doc_metadata] # Must be a list of metadatas (one for each text)
        )

        final_chunks: List[DocumentChunk] = []
        for i, lc_doc in enumerate(langchain_documents):
            # lc_doc.metadata already contains 'original_source', 'start_index', and base_metadata.
            # We'll use source_path for the 'source' field of our DocumentChunk for consistency.
            chunk_specific_metadata = lc_doc.metadata.copy()
            chunk_specific_metadata["chunk_index_in_doc"] = i # Add our own sequential chunk index

            final_chunks.append(DocumentChunk(
                content=lc_doc.page_content,
                source=source_path, # The original document source path
                metadata=chunk_specific_metadata
            ))
        
        print(f"SmartChunker: Document from '{source_path}' split into {len(final_chunks)} chunks.")
        return final_chunks

    def chunk(self, documents_to_chunk: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Takes a list of DocumentChunk objects (where each 'content' field is a full document text)
        and returns a new list of DocumentChunk objects where content has been chunked.
        """
        all_resulting_chunks = []
        for original_doc_chunk in documents_to_chunk:
            if not isinstance(original_doc_chunk, DocumentChunk):
                print(f"Warning: SmartChunker.chunk expected DocumentChunk, got {type(original_doc_chunk)}. Skipping.")
                continue
            
            # The 'metadata' of the input DocumentChunk (representing the full document)
            # will be used as the 'base_metadata' for the smaller chunks created from its content.
            newly_chunked_parts = self._chunk_single_document_content(
                document_content=original_doc_chunk.content,
                source_path=original_doc_chunk.source, # Pass the original source identifier
                base_metadata=original_doc_chunk.metadata # Pass the original document's metadata
            )
            all_resulting_chunks.extend(newly_chunked_parts)
        return all_resulting_chunks
