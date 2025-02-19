from Artemisa.Extractor import ExcelExtractor
from Artemisa.Llm.providers.openai import OpenAIAgent
from api_key import OPENAI_API_KEY
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

excel = ExcelExtractor('Cierre-Octubre.xlsx')

output_file, content =excel.excel()

client = OpenAIAgent(OPENAI_API_KEY)

print('Haciendo query al agente')

query = f'Oye que opinas de los gastos de esta empresa estudiantil, y que mejoras se pueden hacer?, datos: {content}'

response = client.query(query=query, format_response=True)


print_response(response)