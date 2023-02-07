from data import CodeGPTDataset
from model import get_gpt2
import torch
from   torch.utils.data import DataLoader, SequentialSampler
from   beam import Beam


def predict_single(inputs, model, tokenizer, max_gen_len=100):
    with torch.no_grad():
        beam_size = 10
        softmax = torch.nn.LogSoftmax(dim=-1)
        hidden_layers = model(inputs).past_key_values
        p = []
        zero = torch.cuda.LongTensor(1).fill_(0)
        
        past = [
            torch.cat(
                [layer[0].unsqueeze(0), layer[1].unsqueeze(0)],
                dim=0
            )
            if type(layer) == tuple else layer
            for layer in hidden_layers
        ]

        past_hidden = [
            layer.expand(-1, beam_size, -1, -1, -1)
            for layer in past
        ]

        beam = Beam(beam_size, tokenizer.bos_token_id, tokenizer.eos_token_id)
        input_ids = None
        for _ in range(max_gen_len): 
            if beam.done():
                break
            input_ids = beam.getCurrentState()    
            transformer_outputs = model(input_ids, past_key_values=past_hidden)
            out = softmax(transformer_outputs[0][:, -1, :]).data
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
    dataset = CodeGPTDataset.from_test_data(
        datasource=datasource,
        tokenizer=tokenizer
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    model.to(device)
    model.zero_grad()
    model.eval()

    for input_ids, batch, token_labels in dataloader:
        batch_tensor = batch.to(device)
        yield predict_single(batch_tensor[0], model, tokenizer)


if __name__ == "__main__":
    import data.formats
    import transformers

    transformers.logging.set_verbosity_error()

    datasource = (nl for nl, code in data.formats.conala("datasets/conala/test.json"))
    predictions = predict("microsoft/CodeGPT-small-py-adaptedGPT2", datasource)
    prediction = next(predictions)

    print(prediction)