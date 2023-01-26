from util import *
from data.formats import *
from train import train
from predict import predict


if __name__ == "__main__":
    root = output_directory("test")
    datasource = conala("datasets/conala/train.json")
    model = train(
        model="microsoft/CodeGPT-small-py-adaptedGPT2",
        datasource=datasource,
        output_directory=root / f"model",
        n_epochs=1
    )

    print("Model:", model)

    output_source = predict(
        model=model,
        datasource=conala("datasets/conala/test.json")
    )

    save_output(
        output_file=root / "output",
        output_source=output_source
    )