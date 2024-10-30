from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model an tokenizer

MODEL_NAME = "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

print("MODEL LOADED")

# Ensure using GPU if available

print(f"CUDA AVAILABILITY: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare questions

INITIAL_QUESTIONS = ["Is France in the European Union?", "Is Greece a poor country?"]
FOLLOW_UP_QUESTIONS = [
    "Is France known for its wine production?",
    "Does the Eiffel Tower stand in Paris?",
    "Is the French language one of the official languages of the United Nations?",
    "Was the Louvre Museum originally built as a fortress?",
    "Is French cuisine recognized as a UNESCO intangible cultural heritage?",
    "Are the French Alps a popular destination for skiing?",
    "Did France participate in World War I?",
    "Is the currency used in France the Euro?",
    "Is the French flag known as the Tricolore?",
    "Was Napoleon Bonaparte exiled to the island of Elba?",
    "Is the French Revolution celebrated on July 14th?",
    "Does France have a coastline along the Mediterranean Sea?",
    "Is Paris the capital city of France?",
    "Are there more than 500 varieties of cheese produced in France?",
    "Is the Tour de France a famous bicycle race held annually?",
    "Does France share a border with Germany?",
    "Is croissant a traditional French pastry?",
    "Is the Château de Versailles located near Paris?",
    "Are French and English the only official languages in France?",
    "Was Marie Curie the first woman to win a Nobel Prize in France?"
]

# Generate response function

def generate_response(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    attention_mask = torch.ones(input_ids.shape, device=device)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Generate responses and save results

import re
import json
from datetime import datetime

json_dict = {}

for j, initial_question in enumerate(INITIAL_QUESTIONS):
    json_dict[j] = {}
    json_dict[j]["initial_question"] = {}
    json_dict[j]["initial_question"]["original"] = initial_question

    initial_question_input = f"Question: {initial_question}\nAnswer:"
    json_dict[j]["initial_question"]["input"] = initial_question_input

    initial_question_output = generate_response(initial_question_input)
    json_dict[j]["initial_question"]["output"] = initial_question_output

    initial_question_output_deconstructed = initial_question_output.replace(initial_question_input, '').strip()
    json_dict[j]["initial_question"]["output_cleaned"] = initial_question_output_deconstructed

    json_dict[j]["yes_no_array"] = []
    json_dict[j]["follow_up_questions"] = {}

    print(f"FOLLOW UP QUESTION {j + 1} ANSWERED")

    for i, question in enumerate(FOLLOW_UP_QUESTIONS):
        json_dict[j]["follow_up_questions"][i] = {}
        json_dict[j]["follow_up_questions"][i]["original"] = question

        follow_up_question_constructed = f"{initial_question_output}\nFollow-Up Question: {question}\nPlease answer with 'yes' or 'no':"
        json_dict[j]["follow_up_questions"][i]["input"] = follow_up_question_constructed

        follow_up_response = generate_response(follow_up_question_constructed, 10)
        json_dict[j]["follow_up_questions"][i]["output"] = follow_up_response

        follow_up_response_trimmed = follow_up_response.replace(follow_up_question_constructed, '').strip()
        json_dict[j]["follow_up_questions"][i]["output_cleaned"] = follow_up_response_trimmed

        yes_count = len(re.findall(r"\byes\b", follow_up_response_trimmed, flags=re.IGNORECASE))
        no_count = len(re.findall(r"\bno\b", follow_up_response_trimmed, flags=re.IGNORECASE))


        if yes_count > no_count:
            json_dict[j]["follow_up_questions"][i]["yes_or_no"] = "yes"
            json_dict[j]["yes_no_array"].append(2)
        elif yes_count < no_count:
            json_dict[j]["follow_up_questions"][i]["yes_or_no"] = "no"
            json_dict[j]["yes_no_array"].append(0)
        else:
            json_dict[j]["follow_up_questions"][i]["yes_or_no"] = "n/a"
            json_dict[j]["yes_no_array"].append(1)

        print(f"FOLLOW UP QUESTION {j + 1}.{i + 1} ANSWERED")

with open("results.json", "r") as f:
    current_json = json.load(f)

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

current_json[dt_string] = json_dict

with open("results.json", "w") as f:
    json.dump(current_json, f)
