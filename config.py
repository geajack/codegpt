import yaml
from pathlib import Path

from data import CodeGPTDataset
from model import *
from data.datasources import get_datasource


def read_config(path, mode):
    full_path = Path(path)
    with open(full_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    config_name = full_path.stem

    data_format           = config["dataset"]["format"]
    data_file_path        = config["dataset"]["file"]
    datasource_parameters = config["dataset"].get("parameters", {})
    model_source          = config["model"]["source"]
    model_uri             = config["model"]["uri"]
    parameters            = config.get("parameters", {})

    if model_source == "local":
        model_home = Path("../models")
        full_model_uri = str((model_home / model_uri).absolute())
    else:
        full_model_uri = model_uri

    dataset_home = Path("datasets")
    full_dataset_path = str((dataset_home / data_file_path).absolute())

    model, tokenizer = get_gpt2(full_model_uri)

    datasource_function = get_datasource(data_format)
    datasource = datasource_function(full_dataset_path, **datasource_parameters)

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