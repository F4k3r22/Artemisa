# Este es un test de uso de DeepResearcherLocal y LocalSearchEngine que es levantado por Flask en el puerto 5000
import requests
import re

BASE_URL = 'http://127.0.0.1:5000'

def test_api_query():
    response = requests.post(
        f'{BASE_URL}/query',
        json={
            "query": "Oye haz un resumen ejecutivo de todo el contenido que encuentres en mis documentos de relacionados a corrupción, El Salvador, y política. Por favor, incluye las fuentes. Gracias.",
            "fast_mode": False
        }
    )

    print(f"Status Code: {response.status_code}")

    try:
        response_json = response.json()
        raw_response = response_json.get("response", "")
        
        # Extraer el contenido limpio usando regex para obtener el texto dentro de ChatCompletionMessage(content='...')
        content_match = re.search(r"ChatCompletionMessage\(content='(.*?)',\s*refusal=", raw_response, re.DOTALL)
        if content_match:
            clean_content = content_match.group(1).strip()
        else:
            # Si no se encuentra el patrón, se toma la parte anterior a "### Sources:"
            clean_content = raw_response.split("### Sources:")[0].strip()
        
        # Extraer las fuentes después de "### Sources:"
        if "### Sources:" in raw_response:
            sources_raw = raw_response.split("### Sources:")[-1].strip()
        else:
            sources_raw = "No sources found"

        print("\n=== Clean Content ===\n")
        print(clean_content)

        print("\n=== Sources ===\n")
        print(sources_raw)

    except requests.exceptions.JSONDecodeError as e:
        print("Failed to decode JSON response")
        print(e)

# Ejemplo de respuesta:
# === Clean Content ===

#El contenido sobre la corrupción en El Salvador subraya el ambiente político bajo la administración de Nayib Bukele desde 2019, marcado 
#por varias denuncias de falta de transparencia y manejo irregular de fondos públicos. Durante la pandemia de COVID-19, se registraron casos notorios, como la entrega de bonos de $300 a beneficiarios sin criterios claros y la compra de insumos médicos a precios sobrevalorados. Además, el nepotismo es evidente, con un 34% de los cargos de confianza ocupados por familiares de funcionarios.\n\nA pesar de la aprobación de la Ley Anticorrupción en 2025, considerada un avance, críticos argumentan que su implementación parece más un esfuerzo político que un compromiso genuino hacia la transparencia. Las reformas se han visto acompañadas de un estado de excepción que ha permitido 
#detenciones masivas y ha creado un clima de creciente descontento social.\n\nEn 2024, la nueva Ley de Minería Metálica generó protestas 
#y desconfianza, a raíz de decisiones legislativas que favorecían intereses económicos a expensas de la protección ambiental. La falta de debate público y transparencia profundiza la desconfianza en las instituciones. \n\nLas organizaciones de derechos humanos han expresado su preocupación por las violaciones bajo el estado de excepción, donde miles han sido detenidos sin procesos adecuados y en condiciones críticas, lo que refleja una grave crisis de derechos humanos.\n\nFuentes:\n1. Proyecto de Seminario del Complejo Educativo Católico "Guadalupe Carcamo".\n2. Documentos sobre la percepción de corrupción en el gobierno actual.\n3. Investigación sobre la corrupción y la conciencia ciudadana en El Salvador.

# === Sources ===
# * static\docs\SEMI.docx : static\docs\SEMI.docx
#* static\docs\SEMI.docx : static\docs\SEMI.docx
# * static\docs\SEMI.docx : static\docs\SEMI.docx
#* static\docs\SEMI.docx : static\docs\SEMI.docx

def test_api_search_form():
    response = requests.post(
        f'{BASE_URL}/search_form',
        json={
            "query": "Corrupción?",
            "num_results": 5,
        }
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    #test_api_search_form()
    test_api_query()
