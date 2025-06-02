

       
class RAGUpdateHandler:
    def __init__(self, manifest, chroma_collection, embedder=None):
        self.manifest = manifest
        self.chroma_collection = chroma_collection
        self.embedder = embedder

    def process_new_file(self, file_path):
        print(f"Processing new file: {file_path}")
        self.chroma_collection.add_chunks(
            ids=[file_path.name],
            texts=[f"Content of {file_path}"],
            metadatas=[{"source": str(file_path)}],
            embeddings_list=None 
        )
        self.manifest.add_file(file_path, metadata={"source": str(file_path)})

    def process_modified_file(self, file_path):
        print(f"Processing modified file: {file_path}")

    def process_deleted_file(self, file_path):
        print(f"Processing deleted file: {file_path}")
