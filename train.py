import os

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
import numpy as np
import random

from model import get_gpt2
from data import CodeGPTDataset


def set_seed(seed, multiple_gpus=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if multiple_gpus:
        torch.cuda.manual_seed_all(seed)


def train(
    model,
    datasource,
    output_directory,
    per_gpu_train_batch_size=6,
    gradient_accumulation_steps=2,
    max_steps=None,
    n_epochs=30,
    weight_decay=0.01,
    learning_rate=5e-5,
    adam_epsilon=1e-8,
    warmup_steps=0,
    seed=0,
    max_grad_norm=1.0,
    log_every=100,
    save_every=5000
):
    model, tokenizer = get_gpt2(model)
    dataset = CodeGPTDataset.from_training_data(
        datasource=datasource,
        tokenizer=tokenizer
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    batch_size = per_gpu_train_batch_size
    batch_size = batch_size * gradient_accumulation_steps

    train_sampler = RandomSampler(dataset)    
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)
    total_examples = len(dataset)
    
    if max_steps is None:
        max_steps = total_examples // batch_size * n_epochs
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=max_steps)
    checkpoint_last = os.path.join(output_directory, "checkpoint-last")
    scheduler_last = os.path.join(checkpoint_last, "scheduler.pt")
    optimizer_last = os.path.join(checkpoint_last, "optimizer.pt")
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))

    print("Beginning training")
    
    n_weight_updates = 0
    n_batches_per_epoch = len(train_dataloader)
    logging_loss = 0
    average_loss = 0
    tr_loss = 0
    tr_nb = 0
    
    model.zero_grad()
    set_seed(seed)  # Added here for reproducibility (even between python 2 and 3)    
    l = CrossEntropyLoss()

    do_stop = False 
    for epoch_index in range(0, int(n_epochs)): 
        epoch_number = epoch_index + 1
        for batch_index, (input_ids, batch, token_labels) in enumerate(train_dataloader):
            batch_number = batch_index + 1

            inputs = batch.to(device)
            attn_mask = (token_labels.detach() != 0).type(torch.uint8).to(device)
            loss_mask = (token_labels.detach() == 2).type(torch.uint8).to(device)
            
            model.train()
            outputs = model(inputs, attention_mask=attn_mask)
            logits = outputs[0]
            labels = inputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
            ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
            loss = l(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])            
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
                
            if batch_number % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                n_weight_updates += 1

                average_loss = round(np.exp((tr_loss - logging_loss) / (n_weight_updates - tr_nb)), 4)
                if n_weight_updates % log_every == 0:
                    print(f"  steps: {n_weight_updates}  ppl: {round(average_loss, 5)}")
                
                logging_loss = tr_loss
                tr_nb = n_weight_updates

                do_save = False                
                if save_every > 0 and n_weight_updates % save_every == 0:
                    do_save = True
                if epoch_number == n_epochs:
                    index_of_last_weight_update = (n_batches_per_epoch // gradient_accumulation_steps) * gradient_accumulation_steps - 1
                    if batch_index == index_of_last_weight_update:
                        do_save = True

                if max_steps > 0 and n_weight_updates > max_steps:
                    do_save = True
                    do_stop = True

                if do_save:
                    checkpoint_prefix = "checkpoint"                    
                    checkpoint_dir = os.path.join(output_directory, "{}-{}".format(checkpoint_prefix, n_weight_updates))
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    
                    model_to_save = model.module if hasattr(model, "module") else model
                    
                    model_to_save.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    print("Saved model checkpoint to %s", checkpoint_dir)

                    last_output_dir = os.path.join(output_directory, "checkpoint-last")
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    
                    model_to_save.save_pretrained(last_output_dir)
                    tokenizer.save_pretrained(last_output_dir)
                    
                    idx_file = os.path.join(last_output_dir, "idx_file.txt")
                    with open(idx_file, "w", encoding="utf-8") as idxf:
                        idxf.write(str(0) + "\n")

                    torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                    print("Saved optimizer and scheduler states to %s", last_output_dir)

                    step_file = os.path.join(last_output_dir, "step_file.txt")
                    with open(step_file, "w", encoding="utf-8") as stepf:
                        stepf.write(str(n_weight_updates) + "\n")                    

            if do_stop:
                break
        if do_stop:
            break

    return last_output_dir


if __name__ == "__main__":
    from sys import argv
    from pathlib import Path
    from datetime import datetime

    from config import read_config

    config_path = argv[1]
    print("Running train.py", config_path)

    model, tokenizer, dataset, parameters, config_name = read_config(config_path, "train")

    model_home = Path("/home/ICTDOMAIN/d20126116/Code/CodeGPT/models")
    now = datetime.now().strftime("%d-%m-%y@%H:%M:%S")
    output_directory_name = f"{config_name}-{now}"
    output_directory = (model_home / output_directory_name).absolute()

    train(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        output_directory=output_directory,
        **parameters
    )