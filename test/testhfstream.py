from Artemisa.Extractor.excel import ExcelExtractor
from Artemisa.Llm.providers import HugginFaceClient
from api_key import HF_API_KEY

excel = ExcelExtractor('Cierre-Octubre.xlsx')

output_file, content = excel.excel()

print('Iniciando query a HugginFaceClient')
client = HugginFaceClient(HF_API_KEY, 'mistralai/Mistral-7B-Instruct-v0.3', 'hf-inference', max_tokens=4500, stream=True)

query = f'Oye que opinas de los gastos de esta empresa estudiantil, y que mejoras se pueden hacer?, datos: {content}'

# Hay modelos que te van arrojar este error: Bad request:
# Model requires a Pro subscription; check out hf.co/pricing to learn more. Make sure to include your HF token in your query.

for chunk in client.query(query):
    print(chunk, end="")