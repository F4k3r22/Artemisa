from Artemisa.Extractor.excel import ExcelExtractor
from Artemisa.Llm.providers import HuggingFaceClient
from api_key import HF_API_KEY

excel = ExcelExtractor('Cierre-Octubre.xlsx')

output_file, content = excel.excel()

print('Iniciando query a HugginFaceClient')
client = HuggingFaceClient(HF_API_KEY, 'meta-llama/Llama-3.1-8B-Instruct', 'hf-inference')

query = f'Oye que opinas de los gastos de esta empresa estudiantil, y que mejoras se pueden hacer?, datos: {content}'

response = client.query(query=query)

# Hay modelos que te van arrojar este error: Bad request:
# Model requires a Pro subscription; check out hf.co/pricing to learn more. Make sure to include your HF token in your query.

print(response)