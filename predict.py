from data import CodeGPTDataset
from model import get_gpt2
import torch
from   torch.utils.data import DataLoader, SequentialSampler
from   beam import Beam


def predict_single(inputs, model, tokenizer, max_gen_len=100):
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
            pred = beam.buildTargetTokens(hyp)[:beam_size]

            pred = [torch.cat([x.view(-1) for x in p]+[zero]*(max_gen_len-len(p))).view(1,-1) for p in pred]
            p.append(torch.cat(pred, 0).unsqueeze(0))
        p = torch.cat(p, 0)
        for pred in p:
            t = pred[0].cpu().numpy()
            t = list(t)
            if 0 in t:
                t = t[:t.index(0)]
            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
            return text


def predict(model, datasource):
    model, tokenizer = get_gpt2(model)
    dataset = CodeGPTDataset(
        datasource=datasource,
        tokenizer=tokenizer,
        mode="test"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    model.to(device)
    model.zero_grad()
    model.eval()

    for input_ids, batch, token_labels in dataloader:
        yield predict_single(batch.to(device), model, tokenizer)


if __name__ == "__main__":
    from sys import argv
    from pathlib import Path
    from datetime import datetime

    from config import read_config

    config_path = argv[1]
    print("Running predict.py", config_path)

    model, tokenizer, dataset, parameters, config_name = read_config(config_path, "test")

    output_home = Path("output/predictions")
    now = datetime.now().strftime("%d-%m-%y@%H:%M:%S")
    output_directory_name = f"{config_name}-{now}"
    output_filepath = (output_home / output_directory_name).absolute()

    with open(output_filepath, "wb") as output_file:
        for prediction in predict(model, tokenizer, dataset):
            buffer = prediction.encode("utf-8")
            output_file.write(buffer)
            output_file.write(b"\0")
            output_file.flush()
            print(prediction)