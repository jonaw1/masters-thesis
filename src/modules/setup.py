from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

GPT2_XL_MODEL_NAME = "gpt2-xl"
GPT_J_MODEL_NAME = "EleutherAI/gpt-j-6B"


def get_model_tokenizer_device(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device
