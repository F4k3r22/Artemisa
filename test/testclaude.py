from Artemisa.Extractor.excel import ExcelExtractor
from Artemisa.Llm.providers import AnthropicAgent
from api_key import ANTHROPIC_API

excel = ExcelExtractor('Cierre-Octubre.xlsx')

output_file, content = excel.excel()

print('Iniciando query a Anthropic')
client = AnthropicAgent(ANTHROPIC_API)

query = f'Oye que opinas de los gastos de esta empresa estudiantil, y que mejoras se pueden hacer?, datos: {content}'

response = client.query(query=query)


print(response)