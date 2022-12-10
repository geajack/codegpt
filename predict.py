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

    dataset = concodeDataset(
        tokenizer=tokenizer,
        data_dir="datasets/concode",
        cache_file="output/concode_test_preprocessed.pickle",
        logger=logger,
        file_type="test",
        block_size=512,
        mode="test"
    )
    for prediction in predict(model, tokenizer, dataset, device):
        print(prediction)

    # sampler = SequentialSampler(dataset)
    # dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)
    # for index, (batch, token_labels) in enumerate(dataloader):
    #     if index == 88:
    #         break

    # beam_size = 10

    # model.to(device)
    # model.zero_grad()
    # model.eval()

    # beam = Beam(beam_size, tokenizer.bos_token_id, tokenizer.eos_token_id)
    # beam_state = beam.getCurrentState()
    # print("Beam state")
    # print(len(beam_state))
    # print(beam_state[0].shape)
    # print()

    # inputs = batch.to(device)
    # print("Inputs")
    # print(inputs.shape)
    # print()

    # with torch.no_grad():
    #     outputs = model(inputs)[1]
    #     print("Outputs")
    #     print("Tuple of length", len(outputs))
    #     print(outputs[0].shape)
    #     print()
        
    #     past = [torch.cat([x[0].unsqueeze(0),x[1].unsqueeze(0)],dim=0) if type(x)==tuple else x for x in outputs]
    #     past_hidden = [x[:, 0:1].expand(-1, beam_size, -1, -1, -1) for x in past]
    #     print("past", len(past_hidden), past[0].shape, sep=" - ")
    #     print("past_hidden", len(past_hidden), past_hidden[0].shape, sep=" - ")