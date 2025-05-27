

class LlamaCppInterface:

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs


class LocalMemoryInterface:

    def __init__(self, memory_path: str):
        self.memory_path = memory_path

    def load_memory(self):
        pass 




class EmbeddingStoreInterface:
    
    def __init__(self, store_path: str):
        self.store_path = store_path

    def load_embeddings(self):
        pass

    def save_embeddings(self, embeddings):
        pass