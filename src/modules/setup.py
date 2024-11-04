from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

MODEL_NAME = "gpt2-xl"

def get_model_tokenizer_device():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device