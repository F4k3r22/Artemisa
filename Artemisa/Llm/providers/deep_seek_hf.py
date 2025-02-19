from huggingface_hub import InferenceClient

class DeepSeekR1Qwen32B:
    def __init__(self, API_KEY_HF, max_tokens=None, stream=False):
        """ Solo estara disponible mientras HuggingFace 
            de acceso gratuito a deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"""
        
        self.api_key = API_KEY_HF

        self.max_tokens = max_tokens

        if self.max_tokens is None:
            self.max_tokens = 1500

        self.stream = stream

    def queryR1Qwen(self, query: str, format_response=False):
        query = query
        client =  InferenceClient(
	        provider="hf-inference",
	        api_key=self.api_key
        )

        try:

            messages = [
	            {
		            "role": "user",
		            "content": query
	            }
            ]

            completion = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
	            messages=messages, 
	            max_tokens=self.max_tokens,
            )

            if format_response is True:
                return self._format_response(completion)
            else:
                return completion.choices[0].message
            
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
