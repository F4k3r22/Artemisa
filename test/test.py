from Artemisa.Extractor.excel import ExcelExtractor
from Artemisa.Extractor.word import DocxExtractor
from Artemisa.Extractor.pptx import PPTXExtractor

excel = ExcelExtractor('Cierre-Octubre.xlsx')

output_file, content =excel.excel()

print(content)

docs = DocxExtractor('Recursos.docx')

print(docs.extract_all())

# Crear instancia
extractor = PPTXExtractor("presentacion.pptx")

# Extraer todo el contenido
content = extractor.extract_all()

# Acceder a elementos específicos
slides = content['slides']
properties = content['properties']
stats = content['presentation_stats']

# Guardar imágenes
extractor.save_images("imagenes_extraidas")

# Imprimir información de cada diapositiva
for slide in slides:
    print(f"\nDiapositiva {slide['slide_number']}:")
    print(f"Layout: {slide['slide_layout']}")
    
    # Imprimir texto de cada forma
    for shape in slide['shapes']:
        if 'text' in shape:
            print(f"Texto en {shape['shape_name']}: {shape['text']}")
        
        # Imprimir datos de tabla si existe
        if 'table_data' in shape:
            print("Contenido de tabla:")
            for row in shape['table_data']:
                print([cell['text'] for cell in row])