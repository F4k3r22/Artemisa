from Artemisa.Extractor.excel import ExcelExtractor
from Artemisa.Llm.providers import DeepSeekR1Qwen32B
from api_key import HF_API_KEY

excel = ExcelExtractor('Cierre-Octubre.xlsx')

output_file, content = excel.excel()

print('Iniciando query a R1')
client = DeepSeekR1Qwen32B(HF_API_KEY, stream=True)

query = f'Oye que opinas de los gastos de esta empresa estudiantil, y que mejoras se pueden hacer?, datos: {content}'

# La implementación en Stream si funciona y 
# no tiene el error de Timeout de la versión que manda la petición y espera la respuesta

for chunk in client.queryR1Qwen(query):
    print(chunk, end="")