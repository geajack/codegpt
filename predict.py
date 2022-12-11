import sys
import torch
from   torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from   torch.utils.data.distributed import DistributedSampler
from   beam import Beam
import logging

from run     import set_seed, MODEL_CLASSES, update_config
from dataset import concodeDataset


def predict(model, tokenizer, dataset, device, log_every=100):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    model.to(device)
    model.zero_grad()
    model.eval()

    preds = []
    max_gen_len = 100
    for step, (batch, token_labels) in enumerate(dataloader):
        inputs = batch.to(device)
        
        with torch.no_grad():
            beam_size = 10
            m = torch.nn.LogSoftmax(dim=-1)
            outputs = model(inputs)[1]
            p = []       
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(inputs.shape[0]):
                past = [torch.cat([x[0].unsqueeze(0),x[1].unsqueeze(0)],dim=0) if type(x)==tuple else x for x in outputs]
                past_hidden = [x[:, i:i+1].expand(-1, beam_size, -1, -1, -1) for x in past]
                beam = Beam(beam_size, tokenizer.bos_token_id, tokenizer.eos_token_id)
                input_ids = None
                for _ in range(max_gen_len): 
                    if beam.done():
                        break
                    input_ids = beam.getCurrentState()    
                    transformer_outputs = model(input_ids, past=past_hidden)
                    out = m(transformer_outputs[0][:, -1, :]).data
                    beam.advance(out)
                    past = [torch.cat([x[0].unsqueeze(0),x[1].unsqueeze(0)],dim=0) if type(x)==tuple else x for x in transformer_outputs[1]]
                    past_hidden = [x.data.index_select(1, beam.getCurrentOrigin()) for x in past]
                hyp = beam.getHyp(beam.getFinal())
                pred  =beam.buildTargetTokens(hyp)[:beam_size]

                pred = [torch.cat([x.view(-1) for x in p]+[zero]*(max_gen_len-len(p))).view(1,-1) for p in pred]
                p.append(torch.cat(pred, 0).unsqueeze(0))
            p = torch.cat(p, 0)
            for pred in p:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                yield text


if __name__ == "__main__":
    model_type   = "gpt2"
    pretrain_dir = "../models/concode"
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