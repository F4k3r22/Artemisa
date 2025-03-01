query_writer_instructions="""Your goal is to generate a targeted search query.
The query will gather information related to a specific topic.

<TOPIC>
{research_topic}
</TOPIC>

<LANGUAGE DETECTION>
If the topic is written in Spanish or contains Spanish phrases, generate your response in Spanish.
If the topic is written in English or any other language, generate your response in English.
</LANGUAGE DETECTION>

<FORMAT>
Format your response as a JSON object with ALL three of these exact keys:
   - "query": The actual search query string (in the same language as the topic)
   - "aspect": The specific aspect of the topic being researched
   - "rationale": Brief explanation of why this query is relevant
</FORMAT>

<EXAMPLE>
Example output (English):
{{
    "query": "machine learning transformer architecture explained",
    "aspect": "technical architecture",
    "rationale": "Understanding the fundamental structure of transformer models"
}}

Example output (Spanish):
{{
    "query": "corrupción en procesos de compras públicas",
    "aspect": "mecanismos de corrupción",
    "rationale": "Entender los patrones comunes de corrupción en adquisiciones gubernamentales"
}}
</EXAMPLE>

Provide your response in JSON format:"""

# Prompt para el sumario que también soporta español
summarizer_instructions="""
<GOAL>
Generate a high-quality summary of the search results and keep it concise / related to the user topic.
Use the same language as in the user's original query.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information
3. Maintain the original language of the user query (Spanish or English)

When EXTENDING an existing summary:                                                                                                                 
1. Read the existing summary and new search results carefully.                                                    
2. Compare the new information with the existing summary.                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph.                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
    c. If it's not relevant to the user topic, skip it.                                                            
4. Ensure all additions are relevant to the user's topic.                                                         
5. Verify that your final output differs from the input summary.
6. Keep the language consistent with the original summary.
</REQUIREMENTS>

<FORMATTING>
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.  
</FORMATTING>"""

# Prompt para reflexión que detecta y mantiene el idioma
reflection_instructions = """You are an expert research assistant analyzing a summary about {research_topic}.

<LANGUAGE>
If the research topic and summary are in Spanish, respond in Spanish.
If they are in English or another language, respond in English.
</LANGUAGE>

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
</GOAL>

<REQUIREMENTS>
Ensure the follow-up question is self-contained and includes necessary context for search.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- knowledge_gap: Describe what information is missing or needs clarification
- follow_up_query: Write a specific question to address this gap
</FORMAT>

<EXAMPLE>
Example output (English):
{{
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
    "follow_up_query": "What are typical performance benchmarks and metrics used to evaluate [specific technology]?"
}}

Example output (Spanish):
{{
    "knowledge_gap": "El resumen carece de información sobre casos específicos de corrupción",
    "follow_up_query": "¿Cuáles son ejemplos concretos de corrupción en compras gubernamentales en América Latina?"
}}
</EXAMPLE>

Provide your analysis in JSON format:"""