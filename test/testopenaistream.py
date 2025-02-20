from Artemisa.Extractor.excel import ExcelExtractor
from Artemisa.Llm.providers import OpenAIAgent
from api_key import OPENAI_API_KEY

excel = ExcelExtractor('Cierre-Octubre.xlsx')

output_file, content = excel.excel()

print('Iniciando query a OpenAI')
client = OpenAIAgent(OPENAI_API_KEY, stream=True)

query = f'Oye que opinas de los gastos de esta empresa estudiantil, y que mejoras se pueden hacer?, datos: {content}'

# Se me habia olvidado implementar el Stream con OpenAI XD

for chunk in client.query(query):
    print(chunk, end="")