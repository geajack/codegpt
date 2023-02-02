import human_eval.data as human_eval

import ast


def human_eval_contexts():
    problems = human_eval.read_problems()

    for key in problems.keys():
        problem = problems[key]
        code = problem["prompt"]
        yield code


def human_eval_prompts():
    problems = human_eval.read_problems()

    for key in problems.keys():
        problem = problems[key]

        code = problem["prompt"]

        root = ast.parse(code)

        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef):
                if node.name == problem["entry_point"]:
                    function = node
                    break

        for node in function.body:
            if isinstance(node, ast.Expr):
                docstring = node.value.value
                description = docstring.split(">>>")[0]
                description = description.strip()
                description = description.replace("\n", " ")
                description = description.replace("    ", " ")
                description = description.replace("  ", " ")

                yield description


if __name__ == "__main__":
    for description in human_eval_contexts():
        print(description)
        exit()