import ast
import json

from data import CodeGPTDataset
from model import get_gpt2


def contextual_conala_contexts():
    with open("datasets/contextual_conala/train.jsonl") as file:
        lines = file.readlines()

    for line in lines:
        entry = json.loads(line)
        nl = entry["intent"]
        code = entry["snippet"]
        variables = entry["variables"]

        first_three_words = nl.split()[:3]
        function_name = str.join("_", first_three_words)
        function_name.replace("-", "")

        parameter_string = str.join(",", variables)

        final_code = f"def {function_name}({parameter_string}):\n"
        final_code += f'    """{nl}"""\n'
        for line in code.splitlines():
            final_code += "    " + line + "\n"

        yield final_code

if __name__ == "__main__":
    model, tokenizer = get_gpt2("microsoft/CodeGPT-small-py-adaptedGPT2")
    dataset = CodeGPTDataset.from_training_data(
        (("", code) for code in contextual_conala_contexts()),
        tokenizer
    )
    print(len(dataset))