import os
import sys
import time
from googlesearch import search
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from typing import Optional, Any
from langchain_core.runnables import RunnableConfig

def trace_function_factory(start, Total_timeout=30):
    """Create a trace function to timeout request"""
    def trace_function(frame, event, arg):
        if time.time() - start > Total_timeout:
            raise TimeoutError('Website fetching timed out')
        return trace_function
    return trace_function

def fetch_webpage(url, timeout):
    """Fetch the content of a webpage given a URL and a timeout."""
    start = time.time()
    sys.settrace(trace_function_factory(start))
    try:
        print(f"Fetching link: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        paragraphs = soup.find_all('p')
        page_text = ' '.join([para.get_text() for para in paragraphs])
        return url, page_text
    except (requests.exceptions.RequestException, TimeoutError) as e:
        print(f"Error fetching {url}: {e}")
    finally:
        sys.settrace(None)
    return url, None

def SearchEngine(query, num_search=10, search_time_limit=30, fetch_full_page: bool = False):
    """Perform a Google search and parse the content of the top results."""
    urls = search(query, num_results=num_search)
    max_workers = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_webpage, url, search_time_limit): url for url in urls}
        return {url: page_text for future in as_completed(future_to_url) if (url := future.result()[0]) and (page_text := future.result()[1])}
    
    if fetch_full_page:
        try:
            # Try to fetch the full page content using curl
            import urllib.request
            from bs4 import BeautifulSoup

            response = urllib.request.urlopen(url)
            html = response.read()
            soup = BeautifulSoup(html, 'html.parser')
            raw_content = soup.get_text()
            
            return raw_content
                        
        except Exception as e:
            print(f"Warning: Failed to fetch full page content for {url}: {str(e)}")


@dataclass(kw_only=True)
class Configuration:
    max_web_research_loops: int = 3
    local_llm: str = "deepseek-r1"
    search_api = SearchEngine 
    fetch_full_page: bool = False  # Default to False
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
