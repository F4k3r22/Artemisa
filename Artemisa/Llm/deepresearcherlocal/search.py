from Artemisa.Indexer import LocalDocumentIndexer
from typing import Dict, Optional, Any
import os
from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass, fields
from Artemisa import LocalSearchEngine

@dataclass(kw_only=True)
class Configuration:
    max_web_research_loops: int = 3
    local_llm: str = "deepseek-r1"
    search_api = LocalSearchEngine 
    path : str
    local_model: bool = True # Default to True
    Api_key: str = None
    llm_model: str = "gpt-4o-mini"
    provider: str = "openai"
    max_tokens: int = 1000
    sub_provider_hf: str = "hf-inference"
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v is not None})
