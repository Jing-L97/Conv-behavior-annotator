import random

import ollama
from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI

client = OpenAI()

##########################
# model para settings
##########################

nonLLM_lst = ["blenderbot", "DialoGPT", "finetuned"]
LLM_lst = ["llama3", "mistral-nemo", "mistral", "qwen2.5", "phi3.5"]
gpt_lst = ["chatgpt-4o-latest", "gpt-3.5-turbo-0125"]

blender_para = {
    "model_name": "blenderbot",
    "tokenizer_name": "facebook/blenderbot-400M-distill",
    "model_path": "facebook/blenderbot-400M-distill",
}

finetuned_para = {"model_name": "finetuned", "tokenizer_name": "facebook/blenderbot-400M-distill"}

DialoGPT_para = {
    "model_name": "DialoGPT",
    "tokenizer_name": "microsoft/DialoGPT-medium",
    "model_path": "microsoft/DialoGPT-medium",
}

mistral_para = {"model_name": "mistral"}  # mistral 7B
mistral_nemo_para = {"model_name": "mistral-nemo"}  # mistral 12B
qwen_para = {"model_name": "qwen2.5"}  # qwen 7.6B
llama3_para = {"model_name": "llama3"}  # llama3 8B
gpt4_para = {"model_name": "chatgpt-4o-latest"}
gpt3_para = {"model_name": "gpt-3.5-turbo-0125"}
nemotron_para = {"model_name": "nemotron-mini"}
mixtral_para = {"model_name": "mixtral"}
phi3_5_para = {"model_name": "phi3.5"}

model_para_lst = [
    qwen_para,
    blender_para,
    DialoGPT_para,
    mistral_para,
    llama3_para,
    nemotron_para,
    mixtral_para,
    phi3_5_para,
    mistral_nemo_para,
    finetuned_para,
    gpt4_para,
    gpt3_para,
]

role_dict = {"CHI": "ADULT", "ADULT": "CHI"}
len_dict = {"CHI": 6, "ADULT": 50}
sil_dict = {"CHI": 0.315, "ADULT": 0.037}


##########################
# model para settings
##########################
class TextGenerator:
    def __init__(self, model_name: str, tokenizer, model, device="cpu"):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def generate_LLM(self, messages: list) -> str:
        """Generate the results from the conversational history
        input:
          - conversation history
        return:
          - generated results
          - conversation history
        """
        response = ollama.chat(self.model_name, messages=messages)
        message = response["message"]
        messages.append(message)
        return message["content"], messages

    def generate(self, initial_input: str, messages: list, max_token=12, sil_prob=0) -> str:
        """Generate the input from any type of input with a probability `n` that generation is <SILENCE>."""
        # Determine if we generate <SILENCE> based on the probability `sil_prob`
        if random.random() < sil_prob:
            return "<SILENCE>"
        if self.model_name in ["blenderbot", "DialoGPT", "finetuned"]:
            gen = self.generate_ref(initial_input, max_length=max_token)
        elif self.model_name == "qwen":
            gen = self.generate_qwen(initial_input, "system_role", max_new_tokens=max_token)
        # Add the system_role related prompt at the beginning
        elif self.model_name in LLM_lst:
            gen, messages = self.generate_LLM(messages)
        elif self.model_name == "chatgpt-4o-latest":
            gen = self.generate_gpt(messages)
        return gen
