from util import *
from data.humaneval import human_eval_prompts
from predict import predict


if __name__ == "__main__":
    model = "/home/ICTDOMAIN/d20126116/Code/CodeGPT/results/seeded_noisy_conala-22-01-23@22:59:46/models/threshold_1.0/seed_1/checkpoint-last"

    root = output_directory("evaluate_on_humaneval")

    output_source = predict(
        model=model,
        datasource=human_eval_prompts()
    )

    save_output(
        output_file=root / f"output/threshold_1.0_seed_1",
        output_source=output_source
    )