from .config import SearchEngine, Configuration
from .prompts import *
from .state import *
from .utils import *
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from Artemisa.Llm.local import OllamaLocal
from Artemisa.Llm.providers import *
from typing_extensions import Literal
import json

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
        # Intenta acceder al contenido del mensaje
            if hasattr(result, 'content'):
                query = result.content
            elif hasattr(result, 'message'):
                query = result.message
            else:
            # Fallback: convertir a string y esperar que sea un formato válido
                query = str(result)
        except Exception as e:
            print(f"Error procesando resultado: {e}")

    return {"search_query": query}

def WebResearch(state: SummaryState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)

    # Obtener resultados de búsqueda
    search_results = SearchEngine(state.search_query, num_search=3, fetch_full_page=configurable.fetch_full_page)
    
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
        '  ': ' '      # Espacios dobles
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

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["FinalizeSummary", "WebResearch"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "WebResearch"
    else:
        return "FinalizeSummary"
    

builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("GenerateQuery", GenerateQuery)
builder.add_node("WebResearch", WebResearch)
builder.add_node("SummarizeSources", SummarizeSources)
builder.add_node("ReflectOnSummary", ReflectOnSummary)
builder.add_node("FinalizeSummary", FinalizeSummary)

builder.add_edge(START, "GenerateQuery")
builder.add_edge("GenerateQuery", "WebResearch")
builder.add_edge("WebResearch", "SummarizeSources")
builder.add_edge("SummarizeSources", "ReflectOnSummary")
builder.add_conditional_edges("ReflectOnSummary", route_research)
builder.add_edge("FinalizeSummary", END)

DeepResearcher = builder.compile()