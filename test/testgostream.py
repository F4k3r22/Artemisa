from Artemisa.Llm.providers import GoogleAPI
from Artemisa.Extractor.excel import ExcelExtractor
from api_key import GOOGLE_API

excel = ExcelExtractor('Cierre-Octubre.xlsx')

output_file, content = excel.excel()

print('Iniciando query a Gemini')
client = GoogleAPI(GOOGLE_API, stream=True)

query = f'Oye que opinas de los gastos de esta empresa estudiantil, y que mejoras se pueden hacer?, datos: {content}'

for chunk in client.query(query=query):
    print(chunk)
