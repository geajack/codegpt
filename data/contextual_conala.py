import ast
import json

from data import CodeGPTDataset
from model import get_gpt2


def contextual_conala_contexts(filepath):
    with open(filepath) as file:
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
        body = ""
        for line in code.splitlines():
            body += "    " + line + "\n"

        yield final_code, body


if __name__ == "__main__":
    source = contextual_conala_contexts("/home/ICTDOMAIN/d20126116/Code/CodeGPT/Codebase/datasets/contextual_conala/test.jsonl")
    for n in range(3):
        head, body = next(source)
        print(head, end="")
        print(body)