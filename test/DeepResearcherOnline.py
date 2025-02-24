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

# Si pones como "local" : False, tienes que tomar en cuenta que providers y modelos disponibles hay
# Pon la API key del proveedor que vayas a usar en Api_key
# Ejemplos de config
# "llm_model" : "gpt-4o-mini"
# "provider" : "openai"
# "max_tokens" = 1000
# "sub_provider_hf" : "hf-inference" Este solo si vas a usar HuggingFace

# Providers disponible:
# Proveedor / Alias
# OpenAI = openai
# HuggingFace = hf
# deepseek en HuggingFace = deepseek_hf
# Google o gemini = googleapi
# Anthropic = anthropic

result = DeepResearcherOnline.invoke(input=input_data, config=config)


print_response(result['running_summary'])