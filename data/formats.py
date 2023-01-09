import json


def codexglue(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()

    for line in lines:
        data = json.loads(line)
        yield data["nl"], data["code"]


def conala(filepath):
    with open(filepath) as file:
        data = json.load(file)

    for entry in data:
        nl = entry["rewritten_intent"]
        if not nl:
            nl = entry["intent"]
        code = entry["snippet"]
        yield nl, code


def mbpp(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()

    for line in lines:
        data = json.loads(line)
        yield data["text"], data["code"]


def mbpp_normalized(filepath):
    prefices = (
        "Write a python function",
        "Write a function",
    )
    for nl, code in mbpp(filepath):
        for prefix in prefices:
            if nl.startswith(prefix):
                nl = nl[len(prefix):]
        yield nl, code


def get_format(name):
    return globals()[name]