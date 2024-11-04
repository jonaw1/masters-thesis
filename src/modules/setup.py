from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

GPT2_XL_MODEL_NAME = "gpt2-xl"
GPT_J_MODEL_NAME = "EleutherAI/gpt-j-6B"


def get_model_tokenizer_device(model_name):
    if model_name == GPT2_XL_MODEL_NAME:
        model, tokenizer = get_gpt2_xl()
    elif model_name == GPT_J_MODEL_NAME:
        model, tokenizer = get_gptj()
    else:
        raise ValueError("Unknown model name: " + model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device


def get_gpt2_xl():
    return GPT2LMHeadModel.from_pretrained(
        GPT2_XL_MODEL_NAME
    ), GPT2Tokenizer.from_pretrained(GPT2_XL_MODEL_NAME)


def get_gptj():
    return GPT2LMHeadModel.from_pretrained(
        GPT_J_MODEL_NAME
    ), GPT2Tokenizer.from_pretrained(GPT_J_MODEL_NAME)
