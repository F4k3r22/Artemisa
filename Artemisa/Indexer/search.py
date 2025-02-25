from Artemisa.Indexer import LocalDocumentIndexer
from typing import Dict

class LocalSearchEngine:
    def __init__(self, index_path: str):
        self.indexer = LocalDocumentIndexer()
        self.indexer.index_directory(index_path)

    def search(self, query: str, num_search: int = 3) -> Dict[str, str]:
        results = self.indexer.search(query, limit=num_search)
        return {
            doc["path"]: doc["content"] 
            for doc in results
        }