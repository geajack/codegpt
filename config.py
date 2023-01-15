import yaml
from pathlib import Path

from data import CodeGPTDataset
from model import *
from data.formats import get_format


def load_dataset(path):
    full_path = Path(path)
    with open(full_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    data_format           = config["dataset"]["format"]
    datasource_parameters = config["dataset"].get("parameters", {})
    datasource_function = get_format(data_format)
    datasource = datasource_function(**datasource_parameters)

    return datasource


def read_config(path, mode):
    full_path = Path(path)
    with open(full_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    config_name = full_path.stem

    data_format           = config["dataset"]["format"]
    datasource_parameters = config["dataset"].get("parameters", {})
    model_source          = config["model"]["source"]
    model_uri             = config["model"]["uri"]
    parameters            = config.get("parameters", {})

    if model_source == "local":
        model_home = Path("../models")
        full_model_uri = str((model_home / model_uri).absolute())
    else:
        full_model_uri = model_uri

    model, tokenizer = get_gpt2(full_model_uri)

    datasource_function = get_format(data_format)
    datasource = datasource_function(**datasource_parameters)

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