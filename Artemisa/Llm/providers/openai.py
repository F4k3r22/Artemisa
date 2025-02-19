from openai import OpenAI

class OpenAIAgent:
    def __init__(self, API_KEY, llm_model=None, max_tokens=None):
        self.api_key = API_KEY
        if not API_KEY:
            raise ValueError("API_KEY es requerida")
        
        self.llm_model = llm_model
        self.max_tokens = max_tokens

        if self.llm_model is None:
            self.llm_model = 'gpt-4o-mini'
        
        if self.max_tokens is None:
            self.max_tokens = 1500
    
    def query(self, query: str, system_prompt = None, format_response=False):
        query = query
        system_prompt = system_prompt
        client = OpenAI(api_key=self.api_key)
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": query})

            response = client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=self.max_tokens,
                #stream=True,
                #stream_options={"include_usage": True},
            )

            if format_response is True:
                return self._format_response(response)
            
            else:

                return response

        except Exception as e:
            print(f"Error al procesar la consulta: {str(e)}")
            return None
        
    def _format_response(self, response):
        """
        Formatea la respuesta para una mejor visualización
        """
        if response and response.choices:
            content = response.choices[0].message.content
            return content
        return None