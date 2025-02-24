from .prompts import *
from .search import *
from .state import *
from .utils import *
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from Artemisa.Llm.local import OllamaLocal
from typing_extensions import Literal
import json

def GenerateQuery(state: SummaryState, config: RunnableConfig):

    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = OllamaLocal(model=configurable.local_llm, format="json")
    result = llm_json_mode.query(
        f"Generate a query for web search:",
        query_writer_instructions_formatted
        
    )
    query = json.loads(result)

    return {"search_query": query}

def LocalResearch(state: SummaryState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)

    search_local = LocalSearchEngine(configurable.path)
    # Obtener resultados de búsqueda
    search_results = search_local.search(state.search_query, num_search=3)
    
    # Limpiar los resultados si son un diccionario
    if isinstance(search_results, dict):
        cleaned_results = {url: clean_text(content) for url, content in search_results.items()}
    # Si es una lista de resultados
    elif isinstance(search_results, list):
        cleaned_results = [clean_text(result) if isinstance(result, str) else result for result in search_results]
    else:
        # Si es un string
        cleaned_results = clean_text(search_results)

    # Formatear los resultados limpios
    search_str = deduplicate_and_format_sources(cleaned_results, max_tokens_per_source=1000, include_raw_content=True)

    return {
        "sources_gathered": [format_sources(cleaned_results)], 
        "research_loop_count": state.research_loop_count + 1, 
        "web_research_results": [search_str]
    }

def clean_text(text: str) -> str:
    """Limpia y normaliza el texto de la búsqueda"""
    try:
        # Decodificar caracteres especiales
        text = text.encode('latin1').decode('utf-8')
    except:
        try:
            # Si falla el primer método, intentar con otra codificación
            text = text.encode('utf-8').decode('utf-8')
        except:
            pass
    
    # Reemplazar secuencias problemáticas comunes
    replacements = {
        'Â\xa0': ' ',  # Espacio no rompible
        '\xa0': ' ',   # Espacio no rompible
        '\n': ' ',     # Saltos de línea
        '\t': ' ',     # Tabulaciones
        '  ': ' '      # Espacios dobles
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Eliminar espacios múltiples
    while '  ' in text:
        text = text.replace('  ', ' ')
    
    return text.strip()