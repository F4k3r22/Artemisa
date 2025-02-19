from Artemisa.Extractor.excel import ExcelExtractor
from Artemisa.Llm.providers.deep_seek_hf import DeepSeekR1Qwen32B
from api_key import HF_API_KEY

excel = ExcelExtractor('Cierre-Octubre.xlsx')

output_file, content = excel.excel()

print('Iniciando query a R1')
client = DeepSeekR1Qwen32B(HF_API_KEY)

query = f'Oye que opinas de los gastos de esta empresa estudiantil, y que mejoras se pueden hacer?, datos: {content}'

# Por ahora da un error de timeout por la tardanza en responder, voy a integrar despues la respuesta en Stream
# Para evitar ese error

response = client.queryR1Qwen(query=query)

print(response)