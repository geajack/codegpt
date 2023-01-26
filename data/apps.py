import json
from os import listdir
from pathlib import Path


if __name__ == "__main__":
    root = Path("/home/ICTDOMAIN/d20126116/Datasets/APPS/test")

    for folder in listdir(root):
        metadata_file = root / folder / "metadata.json"
        question_file = root / folder / "question.txt"

        with open(metadata_file, "r") as file:
            metadata = json.load(file)

        difficulty = metadata["difficulty"]
        if difficulty == "introductory":
            print(folder, end=" ")

    print()