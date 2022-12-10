import torch
from   torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from   torch.utils.data.distributed import DistributedSampler
import logging

from run     import set_seed, MODEL_CLASSES, update_config, predict
from dataset import concodeDataset
from beam    import Beam


if __name__ == "__main__":
    model_type   = "gpt2"
    pretrain_dir = "microsoft/CodeGPT-small-java-adaptedGPT2"
    local_rank = -1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.addHandler(logging.FileHandler("output/logs"))

    set_seed(0)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(
        pretrain_dir,
        do_lower_case=False,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<|UNKNOWN|>",
        sep_token="concode_elem_sep"
    )
    logger.info(tokenizer.encode("<s> hello world <pad> </s>"))
    model = model_class.from_pretrained(pretrain_dir)
    model.resize_token_embeddings(len(tokenizer))
    update_config(model, tokenizer)
    logger.info(model.config)

    dataset = concodeDataset(
        tokenizer=tokenizer,
        data_dir="datasets/concode",
        cache_directory="output/cache",
        logger=logger,
        file_type="test",
        block_size=1024,
        mode="test"
    )
    predictions = list(predict(model, tokenizer, dataset, device))

    print(predictions[0])