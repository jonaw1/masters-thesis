import re
import argparse

from modules.questions import FOLLOW_UP_QUESTIONS, INITIAL_QUESTIONS
from modules.setup import (
    get_model_tokenizer_device,
    GPT2_XL_MODEL_NAME,
    GPT_J_MODEL_NAME,
)
from modules.functions import (
    generate_response,
    save_results,
    save_initial_question,
    save_follow_up_question,
)


def main(model_name):
    model, tokenizer, device = get_model_tokenizer_device(model_name)

    results = {}
    res_matrix = []

    for idx, iquest in enumerate(INITIAL_QUESTIONS):
        iquest_input = f"Question: {iquest}\nAnswer:"
        iquest_output = generate_response(
            iquest_input, model=model, tokenizer=tokenizer, device=device
        )
        iquest_output_cleaned = iquest_output.replace(iquest_input, "").strip()

        results = save_initial_question(
            results, idx, iquest, iquest_input, iquest_output, iquest_output_cleaned
        )

        for jdx, fuquest in enumerate(FOLLOW_UP_QUESTIONS):
            fuquest_input = f"{iquest_output}\nFollow-Up Question: {fuquest}\nPlease answer with 'yes' or 'no':"
            fuquest_output = generate_response(
                fuquest_input,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=10,
            )
            fuquest_output_cleaned = fuquest_output.replace(fuquest_input, "").strip()

            yes_count = len(
                re.findall(r"\byes\b", fuquest_output_cleaned, flags=re.IGNORECASE)
            )
            no_count = len(
                re.findall(r"\bno\b", fuquest_output_cleaned, flags=re.IGNORECASE)
            )

            results = save_follow_up_question(
                results,
                idx,
                jdx,
                fuquest,
                fuquest_input,
                fuquest_output,
                fuquest_output_cleaned,
                yes_count,
                no_count,
            )

        res_matrix.append(results[idx]["yes_no_array"])

    save_results(results, model_name, is_matrix=False)
    save_results(res_matrix, model_name, is_matrix=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model-based question generation")
    parser.add_argument(
        "--model",
        choices=[GPT2_XL_MODEL_NAME, GPT_J_MODEL_NAME],
        required=True,
        help="Specify the model name to use (either 'gpt2-xl' or 'EleutherAI/gpt-j-6B')",
    )
    args = parser.parse_args()
    main(args.model)
