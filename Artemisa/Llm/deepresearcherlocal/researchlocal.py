from .prompts import *
from .search import *
from .state import *
from .utils import *
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from Artemisa.Llm.local import OllamaLocal
from Artemisa.Llm.providers import *
from typing_extensions import Literal
import json
from Artemisa import LocalSearchEngine

def GenerateQuery(state: SummaryState, config: RunnableConfig):

    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    if configurable.local_model == False:
        if configurable.provider == "openai":
            llm_json_mode = OpenAIAgent(API_KEY=configurable.Api_key, llm_model=configurable.llm_model, max_tokens=configurable.max_tokens)
        elif configurable.provider == "hf":
            llm_json_mode = HuggingFaceClient(API_KEY_HF=configurable.Api_key, model=configurable.llm_model, provider=configurable.sub_provider_hf, max_tokens=configurable.max_tokens)
        elif configurable.provider == "deepseek_hf":
            llm_json_mode = DeepSeekR1Qwen32B(API_KEY_HF=configurable.Api_key, max_tokens=configurable.max_tokens, format="json")
        elif configurable.provider == "googleapi":
            llm_json_mode = GoogleAPI(API_KEY=configurable.Api_key, llm_model=configurable.llm_model, max_tokens=configurable.max_tokens)
        elif configurable.provider == "anthropic":
            llm_json_mode = AnthropicAgent(API_KEY=configurable.Api_key, llm_model=configurable.llm_model, max_tokens=configurable.max_tokens)
    else:
        llm_json_mode = OllamaLocal(model=configurable.local_llm, format="json")
    result = llm_json_mode.query(
        f"Generate a query for web search:",
        query_writer_instructions_formatted
        
    )
    if isinstance(result, str):
        query = json.loads(result)
    else:
    # Si es un objeto ChatCompletionOutputMessage
    # (Asumiendo que tiene un atributo content o message)
        try:
            if isinstance(result, str):
                query_obj = json.loads(result)
            elif hasattr(result, 'content'):
                query_obj = json.loads(result.content)
            elif hasattr(result, 'message'):
                query_obj = json.loads(result.message)
            else:
                query_obj = json.loads(str(result))
        except Exception as e:
            print(f"Error procesando resultado: {e}")
            query_obj = {"query": state.research_topic}
    
        # Extraer solo el string de búsqueda real
        search_query = query_obj.get("query", state.research_topic)
        search_query = clean_text2(search_query)

    print("Generated query:", search_query)

    return {"search_query": search_query}

def LocalResearch(state: SummaryState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)

    # Verificar si el path está definido en la configuración
    if not hasattr(configurable, 'path') or not configurable.path:
        print("ERROR: La ruta de documentos (path) no está definida en la configuración")
        return {
            "sources_gathered": ["Error: No se encontró la ruta de documentos."], 
            "research_loop_count": state.research_loop_count + 1, 
            "web_research_results": ["No se pudieron buscar documentos debido a que la ruta no está configurada."]
        }
    
    print(f"Buscando en ruta: {configurable.path}")
    try:
        search_local = LocalSearchEngine(configurable.path)
        
        if isinstance(state.search_query, dict):
            search_query = state.search_query.get("query", "")
        else:
            search_query = state.search_query
            
        print(f"Consulta de búsqueda: '{search_query}'")
        
        # Obtener resultados de búsqueda
        search_results = search_local.search(search_query, num_search=15, fallback_to_words=True)
        print(f"Resultados de búsqueda encontrados: {len(search_results) if isinstance(search_results, dict) else 'Ninguno'}")
        
        # Si no hay resultados, devolver mensaje claro
        if not search_results or (isinstance(search_results, dict) and len(search_results) == 0):
            print("ADVERTENCIA: No se encontraron documentos para esta consulta")
            return {
                "sources_gathered": [f"No se encontraron documentos para la consulta: '{search_query}'"], 
                "research_loop_count": state.research_loop_count + 1, 
                "web_research_results": [f"No se encontraron documentos relevantes para la consulta: '{search_query}'. Intente con términos más generales o verifique que los documentos estén correctamente indexados."]
            }
        
        # Procesar los resultados
        if isinstance(search_results, dict):
            cleaned_results = {path: clean_text(content) for path, content in search_results.items()}
        elif isinstance(search_results, list):
            cleaned_results = [clean_text(result) if isinstance(result, str) else result for result in search_results]
        else:
            cleaned_results = clean_text(search_results)

        search_str = deduplicate_and_format_sources(cleaned_results, max_tokens_per_source=1000, include_raw_content=True)
        print(f"Resultados formateados con éxito. Longitud: {len(search_str)}")
        
        return {
            "sources_gathered": [format_sources(cleaned_results)], 
            "research_loop_count": state.research_loop_count + 1, 
            "web_research_results": [search_str]
        }
        
    except Exception as e:
        print(f"ERROR en búsqueda local: {str(e)}")
        return {
            "sources_gathered": [f"Error en búsqueda: {str(e)}"], 
            "research_loop_count": state.research_loop_count + 1, 
            "web_research_results": [f"Ocurrió un error durante la búsqueda de documentos: {str(e)}"]
        }

def SummarizeSources(state: SummaryState, config: RunnableConfig):
    """ Summarize the gathered sources """

    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Search Results> \n {most_recent_web_research} \n <New Search Results>"
        )
    else:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Search Results> \n {most_recent_web_research} \n <Search Results>"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    if configurable.local_model == False:
        if configurable.provider == "openai":
            llm = OpenAIAgent(API_KEY=configurable.Api_key, llm_model=configurable.llm_model, max_tokens=configurable.max_tokens,)
        elif configurable.provider == "hf":
            llm = HuggingFaceClient(API_KEY_HF=configurable.Api_key, model=configurable.llm_model, provider=configurable.sub_provider_hf, max_tokens=configurable.max_tokens)
        elif configurable.provider == "deepseek_hf":
            llm = DeepSeekR1Qwen32B(API_KEY_HF=configurable.Api_key, max_tokens=configurable.max_tokens)
        elif configurable.provider == "googleapi":
            llm = GoogleAPI(API_KEY=configurable.Api_key, llm_model=configurable.llm_model, max_tokens=configurable.max_tokens)
        elif configurable.provider == "anthropic":
            llm = AnthropicAgent(API_KEY=configurable.Api_key, llm_model=configurable.llm_model, max_tokens=configurable.max_tokens)
    else:
        llm = OllamaLocal(model=configurable.local_llm,)
    result = llm.query(
        human_message_content,
        summarizer_instructions
        
    )

    running_summary = result

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models
    # It appears very challenging to prompt them out of the responses
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    return {"running_summary": running_summary}

def ReflectOnSummary(state: SummaryState, config: RunnableConfig):
    """ Reflect on the summary and generate a follow-up query """
    configurable = Configuration.from_runnable_config(config)
    if configurable.local_model == False:
        if configurable.provider == "openai":
            llm_json_mode = OpenAIAgent(API_KEY=configurable.Api_key, llm_model=configurable.llm_model, max_tokens=configurable.max_tokens)
        elif configurable.provider == "hf":
            llm_json_mode = HuggingFaceClient(API_KEY_HF=configurable.Api_key, model=configurable.llm_model, provider=configurable.sub_provider_hf, max_tokens=configurable.max_tokens)
        elif configurable.provider == "deepseek_hf":
            llm_json_mode = DeepSeekR1Qwen32B(API_KEY_HF=configurable.Api_key, max_tokens=configurable.max_tokens, format="json")
        elif configurable.provider == "googleapi":
            llm_json_mode = GoogleAPI(API_KEY=configurable.Api_key, llm_model=configurable.llm_model, max_tokens=configurable.max_tokens)
        elif configurable.provider == "anthropic":
            llm_json_mode = AnthropicAgent(API_KEY=configurable.Api_key, llm_model=configurable.llm_model, max_tokens=configurable.max_tokens)
    else:
        llm_json_mode = OllamaLocal(model=configurable.local_llm, format="json")
    
    # Modificar el prompt para ser más específico sobre el formato JSON requerido
    prompt = f"""Generate a follow-up web search query based on the existing knowledge: {state.running_summary}

    IMPORTANT: Respond ONLY with a valid JSON object in this exact format:
    {{
        "follow_up_query": "your query here"
    }}
    
    Research topic: {state.research_topic}
    """
    
    try:
        result = llm_json_mode.query(prompt, reflection_instructions.format(research_topic=state.research_topic))
        
        # Debug: ver qué está devolviendo el LLM
        print("Debug - LLM result:", result)
        
        # Limpiar y validar el JSON
        if hasattr(result, 'choices') and hasattr(result.choices[0], 'message'):
            # Para respuestas de OpenAI
            result = result.choices[0].message.content
        elif hasattr(result, 'content'):
            # Para algunos tipos de respuesta
            result = result.content
        elif isinstance(result, str):
            # Si ya es una cadena
            result = result
        else:
            # Último recurso
            result = str(result)

        result = result.strip()
        if not (result.startswith('{') and result.endswith('}')):
            # Buscar el JSON en la respuesta
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > 0:
                result = result[start:end]
        
        # Intentar parsear el JSON
        try:
            follow_up_query = json.loads(result)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print("Failed to parse:", result)
            return {"search_query": f"Tell me more about {state.research_topic}"}
        
        query = follow_up_query.get('follow_up_query')
        query = clean_text2(query)
        
        if not query:
            return {"search_query": f"Tell me more about {state.research_topic}"}
            
        return {"search_query": query}
        
    except Exception as e:
        print(f"Error in ReflectOnSummary: {e}")
        # Fallback seguro
        return {"search_query": f"Tell me more about {state.research_topic}"}


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
        '  ': ' ',      # Espacios dobles
        '?' : ' ',      # Signos de interrogación
        '¿' : ' ',      # Signos de interrogación
        '!' : ' ',      # Signos de exclamación
        '¡' : ' ',      # Signos de exclamación
        ':' : ' '      # Dos puntos
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Eliminar espacios múltiples
    while '  ' in text:
        text = text.replace('  ', ' ')
    
    return text.strip()

def clean_text2(text: str) -> str:
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
        '  ': ' ',      # Espacios dobles
        '?' : ' ',      # Signos de interrogación
        '¿' : ' ',      # Signos de interrogación
        '!' : ' ',      # Signos de exclamación
        '¡' : ' ',      # Signos de exclamación
        ':' : ' ',      # Dos puntos
        '.' : ' ',      # Puntos
        ',' : ' ',      # Comas
        ';' : ' ',      # Puntos y comas
        '(' : ' ',      # Paréntesis
        ')' : ' ',      # Paréntesis
        '[' : ' ',      # Corchetes
        ']' : ' ',      # Corchetes
        '{' : ' ',      # Llaves
        '}' : ' ',      # Llaves
        '...' : ' ',    # Puntos suspensivos
        ',': ' ',       # Comas
        '-': ' ',       # Guiones
        '_': ' ',       # Guiones bajos
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Eliminar espacios múltiples
    while '  ' in text:
        text = text.replace('  ', ' ')
    
    return text.strip()

def FinalizeSummary(state: SummaryState):
    """ Finalize the summary """
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["FinalizeSummary", "LocalResearch"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "LocalResearch"
    else:
        return "FinalizeSummary"
    
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("GenerateQuery", GenerateQuery)
builder.add_node("LocalResearch", LocalResearch)
builder.add_node("SummarizeSources", SummarizeSources)
builder.add_node("ReflectOnSummary", ReflectOnSummary)
builder.add_node("FinalizeSummary", FinalizeSummary)

builder.add_edge(START, "GenerateQuery")
builder.add_edge("GenerateQuery", "LocalResearch")
builder.add_edge("LocalResearch", "SummarizeSources")
builder.add_edge("SummarizeSources", "ReflectOnSummary")
builder.add_conditional_edges("ReflectOnSummary", route_research)
builder.add_edge("FinalizeSummary", END)

DeepResearcherLocal = builder.compile()