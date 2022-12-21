import yaml
from pathlib import Path

from data import *
from model import *


def read_config(path, mode):
    full_path = Path(path)
    with open(full_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    config_name = full_path.stem

    data_format    = config["dataset"]["format"]
    data_file_path = config["dataset"]["file"]
    model_source   = config["model"]["source"]
    model_uri      = config["model"]["uri"]
    parameters     = config.get("parameters", {})

    if model_source == "local":
        model_home = Path("../models")
        full_model_uri = str((model_home / model_uri).absolute())
    else:
        full_model_uri = model_uri

    dataset_home = Path("datasets")
    full_dataset_path = str((dataset_home / data_file_path).absolute())

    model, tokenizer = get_gpt2(full_model_uri)

    datasource_types = {
        "codexglue"       : codexglue_datasource,
        "conala"          : conala_datasource,
        "mbpp"            : mbpp_datasource,
        "mbpp-normalized" : mbpp_normalized_datasource
    }
    datasource = datasource_types[data_format](full_dataset_path)

    dataset_parameter_names = ["block_size"]
    dataset_parameters = {}
    for name in dataset_parameter_names:
        if name in parameters:
            dataset_parameters[name] = parameters.pop(name)

    dataset = CodeGPTDataset(
        datasource=datasource,
        tokenizer=tokenizer,
        mode=mode,
        **dataset_parameters
    )

    return model, tokenizer, dataset, parameters, config_name