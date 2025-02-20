from Artemisa.Extractor.excel import ExcelExtractor
from Artemisa.Llm.local import OllamaLocal

excel = ExcelExtractor('Cierre-Octubre.xlsx')

output_file, content = excel.excel()

# Antes de ejecutar con el modelo que quieres debes hacer un ollama pull <modelo> para descargarlo

ollama = OllamaLocal('deepseek-r1', stream=False)

print('Iniciando query a Ollama')

# Tienes que instalar curl -fsSL https://ollama.com/install.sh | sh antes de ejecutarlo
query = f'Oye que opinas de los gastos de esta empresa estudiantil, y que mejoras se pueden hacer?, datos: {content}'

response = ollama.query(query=query)

# Si quieres hacer un stream de la respuesta
#for chunk in ollama.query(query=query):
#    print(chunk)

print(response)