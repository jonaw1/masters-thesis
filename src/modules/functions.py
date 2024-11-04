import json
import datetime

import torch


def generate_response(prompt, model, tokenizer, device, max_length=100) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    attention_mask = torch.ones(input_ids.shape, device=device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def append_results(path, results):
    with open(path, "r", encoding="utf-8") as f:
        current_results = json.load(f)

    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    current_results[dt_string] = results

    with open(path, "w", encoding="utf-8") as f:
        json.dump(current_results, f)


def save_initial_question(
    results, idx, iquest, iquest_input, iquest_output, iquest_output_cleaned
):
    results[idx] = {}
    results[idx]["initial_question"] = {}
    results[idx]["initial_question"]["original"] = iquest

    results[idx]["initial_question"]["input"] = iquest_input

    results[idx]["initial_question"]["output"] = iquest_output

    results[idx]["initial_question"]["output_cleaned"] = iquest_output_cleaned

    results[idx]["yes_no_array"] = []
    results[idx]["follow_up_questions"] = {}

    return results


def save_follow_up_question(
    results,
    idx,
    jdx,
    fuquest,
    fuquest_input,
    fuquest_output,
    fuquest_output_cleaned,
    yes_count,
    no_count,
):
    results[idx]["follow_up_questions"][jdx] = {}
    results[idx]["follow_up_questions"][jdx]["original"] = fuquest

    results[idx]["follow_up_questions"][jdx]["input"] = fuquest_input

    results[idx]["follow_up_questions"][jdx]["output"] = fuquest_output
    results[idx]["follow_up_questions"][jdx]["output_cleaned"] = fuquest_output_cleaned

    if yes_count > no_count:
        results[idx]["follow_up_questions"][jdx]["yes_or_no"] = "yes"
        results[idx]["yes_no_array"].append(2)
    elif yes_count < no_count:
        results[idx]["follow_up_questions"][jdx]["yes_or_no"] = "no"
        results[idx]["yes_no_array"].append(0)
    else:
        results[idx]["follow_up_questions"][jdx]["yes_or_no"] = "n/a"
        results[idx]["yes_no_array"].append(1)

    return results
