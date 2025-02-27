def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=False):
    # Si es un string, formatearlo de manera consistente con el resto de la salida
    if isinstance(search_response, str):
        return f"Sources:\n\nSource Content:\n===\nContent: {search_response}\n===\n"

    # Convert input to list of results
    if isinstance(search_response, dict):
        # For LocalSearchEngine results, convert the dict to a list of dicts with path as url
        sources_list = []
        for path, content in search_response.items():
            sources_list.append({
                'url': path,  # Use path as url
                'title': path.split('/')[-1],  # Use filename as title
                'content': content,
                'raw_content': content
            })
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with paths as keys, a list of search results, or a string")

    # Deduplicate by path/URL
    unique_sources = {}
    for source in sources_list:
        # Si es un string, crear un diccionario con formato similar
        if isinstance(source, str):
            source_dict = {
                'url': 'N/A',
                'title': 'Text Content',
                'content': source,
                'raw_content': source
            }
            if 'N/A' not in unique_sources:  # Evitar duplicados de strings
                unique_sources['N/A'] = source_dict
        # Si es un diccionario, procesar normalmente
        elif isinstance(source, dict) and 'url' in source:
            if source['url'] not in unique_sources:
                unique_sources[source['url']] = source
        else:
            print(f"Warning: Skipping invalid source format: {source}")

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"Path: {source['url']}\n===\n"  # Changed URL to Path
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            char_limit = max_tokens_per_source * 4
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()

def format_sources(search_results):
    """Format search results into a bullet-point list of sources.
    
    Args:
        search_results: Puede ser un diccionario con rutas como claves, una lista de resultados, o un string
        
    Returns:
        str: Formatted string with sources and their paths
    """
    # Si es un string
    if isinstance(search_results, str):
        return f"* Content: {search_results}"
        
    # Si es un diccionario o lista
    try:
        formatted_sources = []
        # Si es un diccionario con rutas como claves (resultados de LocalSearchEngine)
        if isinstance(search_results, dict) and not 'results' in search_results:
            for path, content in search_results.items():
                filename = path.split('/')[-1]
                formatted_sources.append(f"* {filename} : {path}")
        # Si es una lista
        elif isinstance(search_results, list):
            sources = search_results
            # Procesar cada fuente en la lista
            for source in sources:
                if isinstance(source, dict) and 'title' in source and 'url' in source:
                    formatted_sources.append(f"* {source['title']} : {source['url']}")
                elif isinstance(source, str):
                    formatted_sources.append(f"* Content: {source}")
                else:
                    formatted_sources.append(f"* Source: {str(source)}")
        # Si es un diccionario con 'results'
        elif isinstance(search_results, dict) and 'results' in search_results:
            sources = search_results.get('results', [])
            for source in sources:
                if isinstance(source, dict) and 'title' in source and 'url' in source:
                    formatted_sources.append(f"* {source['title']} : {source['url']}")
                else:
                    formatted_sources.append(f"* Source: {str(source)}")
        else:
            return f"* Invalid format: {str(search_results)}"
            
        return '\n'.join(formatted_sources)
        
    except Exception as e:
        print(f"Warning: Error formatting sources: {e}")
        return f"* Error formatting source: {str(search_results)}"