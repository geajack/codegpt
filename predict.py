from data import *
from model import get_gpt2, get_gpt2_tokenizer
import torch
from   torch.utils.data import DataLoader, SequentialSampler
from   beam import Beam


def post_process_hidden_layers(layers):
    processed_layers = []
    for layer in layers:
        if type(layer) == tuple:
            keys, values = layer
            processed = torch.cat([keys.unsqueeze(0), values.unsqueeze(0)], dim=0)
        else:
            processed = layer
        processed_layers.append(processed)
    return processed_layers


def predict_single(inputs, model, tokenizer, max_gen_len=100):
    with torch.no_grad():
        beam_size = 10
        softmax = torch.nn.LogSoftmax(dim=-1)
        hidden_layers = model(inputs).past_key_values
        predictions = []
        zero = torch.cuda.LongTensor(1).fill_(0)
        
        past = post_process_hidden_layers(hidden_layers)
        past_hidden = [
            layer.expand(-1, beam_size, -1, -1, -1)
            for layer in past
        ]

        beam = Beam(beam_size, tokenizer.bos_token_id, tokenizer.eos_token_id)
        for _ in range(max_gen_len): 
            if beam.done():
                break

            input_ids = beam.getCurrentState()
            transformer_outputs = model(input_ids, past_key_values=past_hidden)
            p = softmax(transformer_outputs.logits[:, -1, :]).data
            beam.advance(p)
            past = post_process_hidden_layers(transformer_outputs.past_key_values)
            origin = beam.getCurrentOrigin()
            past_hidden = [x.data.index_select(1, origin) for x in past]
        
        hyp = beam.getHyp(beam.getFinal())
        pred = beam.buildTargetTokens(hyp)[:beam_size]

        pred = [torch.cat([x.view(-1) for x in p]+[zero]*(max_gen_len-len(p))).view(1,-1) for p in pred]
        predictions.append(torch.cat(pred, 0).unsqueeze(0))
        
        predictions = torch.cat(predictions, 0)
        for pred in predictions:
            t = pred[0].cpu().numpy()
            t = list(t)
            if 0 in t:
                t = t[:t.index(0)]
            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
            return text


def predict(model, dataset):
    model, tokenizer = get_gpt2(model)    

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

    model_uri = "microsoft/CodeGPT-small-py-adaptedGPT2"

    tokenizer = get_gpt2_tokenizer(model_uri)

    datasource = data.formats.conala("datasets/conala/test.json")
    dataset = CodeGPTDataset.from_preprocessed(
        (preprocess(nl, "", tokenizer, padding=False) for nl, code in datasource),
        tokenizer=tokenizer
    )
    predictions = predict(model_uri, dataset)
    prediction = next(predictions)

    print(prediction)