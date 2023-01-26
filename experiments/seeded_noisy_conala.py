import datetime
from pathlib import Path

from util import *
from data.formats import *
from train import train
from predict import predict


if __name__ == "__main__":
    root = output_directory("seeded_noisy_conala")

    for threshold in [1.0, 0.6, 0.5, 0.4, 0.3]:
        for seed in [1, 2]:
            print(f"Beginning threshold={threshold}, seed={seed}")

            datasource = conala_noisy(
                mined_filepath="datasets/conala/mined.jsonl",
                train_filepath="datasets/conala/train.json",
                threshold=threshold
            )

            print("Beginning training")

            model = train(
                model="microsoft/CodeGPT-small-py-adaptedGPT2",
                datasource=datasource,
                output_directory=root / f"models/threshold_{threshold}/seed_{seed}",
                seed=seed
            )

            print("Beginning prediction")

            output_source = predict(
                model=model,
                datasource=conala("datasets/conala/test.json")
            )

            save_output(
                output_file=root / f"output/threshold_{threshold}_seed_{seed}",
                output_source=output_source
            )

            print("Finished")

    print("Finished completely")