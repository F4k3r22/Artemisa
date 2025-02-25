from Artemisa import LocalSearchEngine
import os

def test_local_search_engine(path, query, num_results):
    LSE = LocalSearchEngine(path)
    results = LSE.search(query, num_results)
    
    # Mostrar resultados de forma organizada
    print(f"\n=== Resultados para la búsqueda: '{query}' ===\n")
    
    if not results:
        print("No se encontraron resultados.")
        return results
    
    for i, (file_path, content) in enumerate(results.items(), 1):
        # Obtener nombre de archivo y extensión
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].upper()[1:]
        
        # Cabecera del resultado
        print(f"RESULTADO #{i} - {file_name} ({file_ext})")
        print("-" * 80)
        
        # Encontrar líneas relevantes que contienen la consulta
        lines = content.split('\n')
        relevant_lines = []
        context_lines = 2  # Líneas de contexto antes y después
        
        for j, line in enumerate(lines):
            if any(term.lower() in line.lower() for term in query.split()):
                # Agregar líneas de contexto
                start = max(0, j - context_lines)
                end = min(len(lines), j + context_lines + 1)
                
                # Agregar líneas con contexto y marcar la línea con la coincidencia
                for k in range(start, end):
                    if k == j:  # Esta es la línea con la coincidencia
                        relevant_lines.append(f">>> {lines[k]}")
                    else:
                        relevant_lines.append(f"    {lines[k]}")
                
                relevant_lines.append("")  # Espacio entre grupos de coincidencias
        
        # Si no hay líneas relevantes, mostrar las primeras líneas
        if not relevant_lines:
            print("(No se encontraron coincidencias exactas en el texto)")
            preview_lines = 5
            print("\nVista previa:")
            for line in lines[:preview_lines]:
                print(f"    {line}")
            if len(lines) > preview_lines:
                print("    ...")
        else:
            # Mostrar líneas relevantes
            for line in relevant_lines[:15]:  # Limitar a 15 líneas para no saturar
                print(line)
            
            if len(relevant_lines) > 15:
                print("    ...")
        
        print("\n" + "=" * 80 + "\n")
    
    return results

# Ejecutar la búsqueda
Test_local = test_local_search_engine('testfolder', 'Corrupción', 5)