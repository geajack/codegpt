from util import *
from data.humaneval import human_eval_contexts
from predict import predict


if __name__ == "__main__":
    model = "/home/ICTDOMAIN/d20126116/Code/CodeGPT/results/contextual_conala-02-02-23@11:45:53/model/checkpoint-last"

    root = output_directory("evaluate_on_humaneval_contextual_conala")

    output_source = predict(
        model=model,
        datasource=human_eval_contexts()
    )

    save_output(
        output_file=root / f"output",
        output_source=output_source
    )