from google import genai
from google.genai import types

class GoogleAPI:
    def __init__(self, API_KEY, llm_model=None, max_tokens=None, stream=False):

        self.api_key = API_KEY
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.stream = stream

        if self.api_key is None:
            raise Exception('No has proporcionado la API_KEY')

        if self.llm_model is None:
            self.llm_model = 'gemini-2.0-flash'

        if self.max_tokens is None:
            self.max_tokens = 1500
        
    def query(self, query: str, system_prompt = None):

        if self.stream:
            return self.queryStream(query, system_prompt)
        
        query = query
        system_prompt = system_prompt
        client = genai.Client(api_key=self.api_key)
        try:
            # La configuración de la API de Google es rara
            response = client.models.generate_content(
                model = self.llm_model,
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=self.max_tokens
                )
            )

            return response.text
        
        except Exception as e:
            print(f"Error al procesar la consulta: {str(e)}")
            return None
        
    def queryStream(self, query: str, system_prompt = None):
        query = query
        system_prompt = system_prompt
        client = genai.Client(api_key=self.api_key)
        try: 
            response = client.models.generate_content_stream(
                model=self.llm_model,
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=self.max_tokens
                )
            )

            for chunk in response:
                if chunk.text is not None:
                    yield chunk.text

        except Exception as e:
            print(f"Error en el streaming: {str(e)}")
            yield None