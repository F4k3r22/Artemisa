from huggingface_hub import InferenceClient

class DeepSeekR1Qwen32B:
    def __init__(self, API_KEY_HF, max_tokens=None, stream=False, format=None):
        self.api_key = API_KEY_HF
        self.max_tokens = max_tokens if max_tokens is not None else 1500
        self.stream = stream
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.api_key
        )
        self.format = format

    def queryR1Qwen(self, query: str):
        # Si stream está activado, usamos el método de streaming
        if self.stream:
            return self.queryR1QwenStream(query)
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": query
                }
            ]

            completion = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
                messages=messages, 
                max_tokens=self.max_tokens,
                response_format=self.format
            )

            
            return completion.choices[0].message
            
        except Exception as e:
            print(f"Error al procesar la consulta: {str(e)}")
            return None

    def queryR1QwenStream(self, query: str):
        """
        Realiza una consulta al modelo usando streaming
        
        Args:
            query (str): La consulta a realizar
            
        Yields:
            str: Fragmentos de la respuesta
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": query
                }
            ]

            stream = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
                messages=messages, 
                max_tokens=self.max_tokens,
                stream=True,
                response_format=self.format
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"Error en el streaming: {str(e)}")
            yield None