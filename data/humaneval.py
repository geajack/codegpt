import human_eval.data as human_eval

import ast


if __name__ == "__main__":
    problems = human_eval.read_problems()

    keys = list(iter(problems.keys()))
    problem = problems[keys[4]]

    code = problem["prompt"]
    # print(code)

    root = ast.parse(code)

    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef):
            if node.name == problem["entry_point"]:
                function = node
                break

    for node in ast.walk(function):
        if isinstance(node, ast.Expr):
            docstring = node.value.value
            description = docstring.split(">>>")[0]
            description = description.strip()
            description = description.replace("\n", " ")
            description = description.replace("    ", " ")
            description = description.replace("  ", " ")
            print(description)