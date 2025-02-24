from Artemisa.Llm import DeepResearcherOnline
from rich.console import Console
from rich.markdown import Markdown
from api_key import OPENAI_API_KEY

def print_response(response):
    """
    Imprime la respuesta con formato markdown usando rich
    """
    console = Console()
    if response:
        markdown = Markdown(response)
        console.print("\n=== Respuesta del Agente ===\n")
        console.print(markdown)
        console.print("\n===========================\n")

input_data = {
    "research_topic": "Oye haz una investigación completa sobre esta: CORRUPCIÓN DEL ESTADO Y LA CONCIENTIZACIÓN DEL PUEBLO en El Salvador entre el tiempo del 2023 al 2025"
}

# Configuración ajustada
config = { "configurable": { "max_web_research_loops": 5, "fetch_full_page": True, "local_model": False, "Api_key": OPENAI_API_KEY, "llm_model": "gpt-4o-mini", "provider": "openai", "max_tokens": 3000 } }


result = DeepResearcherOnline.invoke(input=input_data, config=config)

print_response(result['running_summary'])