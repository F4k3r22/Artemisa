from Artemisa.Llm.providers import GoogleAPI
from Artemisa.Extractor.excel import ExcelExtractor
from api_key import GOOGLE_API
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

output_file, content = excel.excel()

print('Iniciando query a Gemini')
client = GoogleAPI(GOOGLE_API)

query = f'Oye que opinas de los gastos de esta empresa estudiantil, y que mejoras se pueden hacer?, datos: {content}'

response = client.query(query=query)

print_response(response)