from Artemisa.Llm import DeepResearcherOnline
from rich.console import Console
from rich.markdown import Markdown

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
    "research_topic": "¿Cuál es el impacto de la inteligencia artificial en la medicina moderna?"
}

# Recuerda instalar Ollama y hacer el respetivo ollama pull al model que se va a utilizar

config = {
    "max_web_research_loops": 5,  # Modificar el número de loops
    "local_llm": "deepseek-r1:14b",       # Modificar el modelo local
    "fetch_full_page": True       # Activar la descarga completa de páginas
}


result = DeepResearcherOnline.invoke(input=input_data, config=config)


print_response(result['running_summary'])