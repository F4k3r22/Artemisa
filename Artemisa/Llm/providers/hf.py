from huggingface_hub import InferenceClient

class HugginFaceClient:
    def __init__(self, API_KEY_HF, model, provider, max_tokens=None, stream=False):

        self.api_key = API_KEY_HF
        self.model = model
        self.max_tokens = max_tokens if max_tokens is not None else 1500
        self.stream = stream
        self.provider = provider
        self.client = InferenceClient(
            provider=self.provider,
            api_key=self.api_key
        )

    def query(self, query: str, system_prompt: str):

        if self.stream:
            return self.queryStream(query, system_prompt)

        query = query
        system_prompt = system_prompt
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": query})

            completion = self.client.chat.completions.create(
                model=self.model, 
                messages=messages, 
                max_tokens=self.max_tokens,
            )

            return completion.choices[0].message

        except Exception as e:
            print(f"Error al procesar la consulta: {str(e)}")
            return None
        
    def queryStream(self, query: str, system_prompt: str):
        query = query
        system_prompt = system_prompt
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": query})

            stream = self.client.chat.completions.create(
                model=self.model, 
                messages=messages, 
                max_tokens=self.max_tokens,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"Error en el streaming: {str(e)}")
            return None