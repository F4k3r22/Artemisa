from Artemisa.Extractor import PDFExtractor

# Crear instancia
extractor = PDFExtractor("Documento.pdf")

# Extraer todo el contenido
try:
    content = extractor.extract_all()
    
    # Acceder a la información
    print(f"Total de páginas: {content['total_pages']}")
    print(f"Metadatos: {content['metadata']}")
    
    # Imprimir texto de cada página
    for page in content['pages']:
        print(f"\nPágina {page['page_number']}:")
        print(f"Texto: {page['text'][:200]}...")  # Primeros 200 caracteres
        
        if page['tables']:
            print(f"Tablas encontradas: {len(page['tables'])}")
            
        if page['images']:
            print(f"Imágenes encontradas: {len(page['images'])}")
            
    # Imprimir estadísticas
    print("\nEstadísticas del documento:")
    for key, value in content['text_statistics'].items():
        print(f"{key}: {value}")

except Exception as e:
    print(f"Error procesando el PDF: {e}")