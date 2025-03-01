from Artemisa.Indexer import LocalDocumentIndexer
from typing import Dict

class LocalSearchEngine:
    def __init__(self, index_path: str):
        self.indexer = LocalDocumentIndexer()
        self.indexer.index_directory(index_path)

    def search(self, query: str, num_search: int = 3, fallback_to_words = False) -> Dict[str, str]:
        results = self.indexer.search(query, num_search, fallback_to_words)
        return {
            doc["path"]: doc["content"] 
            for doc in results
        }