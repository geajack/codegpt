from sys import argv
from data.humaneval import human_eval_contexts

from util import read_output_file


if __name__ == "__main__":
    predictions_file = argv[1]
    predictions = read_output_file(predictions_file)
    contexts = human_eval_contexts()
    for prediction, context in zip(predictions, contexts):
        print(context)
        print("    " + prediction)
        print("-" * 200)