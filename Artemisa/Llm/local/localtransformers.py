from transformers import AutoTokenizer, AutoModelForCausalLM

class TransformersLocal:
    def __init__(self, model, tokenizer = None, max_tokens=None):

        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        if self.tokenizer is None:
            self.tokenizer = model

        if self.max_tokens  is None:
            self.max_tokens = 1500

    def queryModel(self, query: str, system_prompt = None):
        """ Acepta modelos tipo no Razonador que permite usar SystemPrompt """
        pass

    def queryModelR1(self, query: str):
        """ Acepta modelo razonador tipo R1 que no permite usar SystemPrompt """
        pass