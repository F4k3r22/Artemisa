from anthropic import Anthropic


class AnthropicAgent:
    def __init__(self, API_KEY, llm_model=None, max_tokens=None, stream=False):
        self.api_key = API_KEY
        if not API_KEY:
            raise ValueError("API_KEY es requerida")
        
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.stream = stream

        if self.llm_model is None:
            self.llm_model = 'claude-3-5-sonnet-latest'
        
        if self.max_tokens is None:
            self.max_tokens = 1500

    def query(self, query: str, system_prompt = None):
        
        query = query
        system_prompt = system_prompt
        client = Anthropic(api_key=self.api_key)
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": query})

            response = client.messages.create(
                max_tokens=self.max_tokens,
                messages=messages,
                model=self.llm_model
            )

            return response.content

        except Exception as e:
            print(f"Error al procesar la consulta: {str(e)}")
            return None
        
    def queryStream(self, query: str, system_prompt = None):

        query = query
        system_prompt = system_prompt
        client = Anthropic(api_key=self.api_key)
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": query})

            stream = client.messages.create(
                max_tokens=self.max_tokens,
                messages=messages,
                model=self.llm_model,
                stream=True
            )

            for chunk in stream:
                if chunk.type is not None:
                    yield chunk.type

        except Exception as e:
            print(f"Error en el streaming: {str(e)}")
            yield None