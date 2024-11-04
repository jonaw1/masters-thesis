from src.modules.questions import FOLLOW_UP_QUESTIONS, INITIAL_QUESTIONS
from src.modules.setup import get_model_tokenizer_device
from src.modules.functions import generate_response, append_results, save_initial_question, save_follow_up_question
import re

model, tokenizer, device = get_model_tokenizer_device()

results = {}

for idx, iquest in enumerate(INITIAL_QUESTIONS):
    iquest_input = f"Question: {iquest}\nAnswer:"
    iquest_output = generate_response(iquest_input)
    iquest_output_cleaned = iquest_output.replace(iquest_input, '').strip()

    results = save_initial_question(results, idx, iquest, iquest_input, iquest_output, iquest_output_cleaned)

    for jdx, fuquest in enumerate(FOLLOW_UP_QUESTIONS):
        fuquest_input = f"{iquest_output}\nFollow-Up Question: {fuquest}\nPlease answer with 'yes' or 'no':"
        fuquest_output = generate_response(fuquest_input, 10)
        fuquest_output_cleaned = fuquest_output.replace(fuquest_input, '').strip()

        yes_count = len(re.findall(r"\byes\b", fuquest_output_cleaned, flags=re.IGNORECASE))
        no_count = len(re.findall(r"\bno\b", fuquest_output_cleaned, flags=re.IGNORECASE))

        results = save_follow_up_question(results, idx, jdx, fuquest, fuquest_input, fuquest_output, fuquest_output_cleaned, yes_count, no_count)


append_results("results.json", results)
