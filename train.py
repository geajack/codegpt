import os

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
import numpy as np

from run import set_seed
from model import get_gpt2
from dataset import CodeGPTDataset


def train(
    train_dataset,
    model,
    tokenizer,
    output_dir,
    tensorboard_dir=None,
    device="cuda",
    batch_size=512,
    local_rank=-1,
    per_gpu_train_batch_size=6,
    n_gpu=1,
    gradient_accumulation_steps=2,
    max_steps=-1,
    num_train_epochs=30,
    weight_decay=0.01,
    learning_rate=5e-5,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gpu_per_node=0,
    seed=0,
    max_grad_norm=1.0,
    logging_steps=100,
    save_steps=5000
):
    """ Train the model """
    if local_rank in [-1, 0]:
        tensorboard_dir = os.path.join(output_dir, 'tensorboard')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        tb_writer = SummaryWriter(tensorboard_dir)
    
    batch_size = per_gpu_train_batch_size * max(1, n_gpu)
    train_sampler = RandomSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)
    total_examples = len(train_dataset) * (
                    torch.distributed.get_world_size() if local_rank != -1 else 1)
    batch_size = batch_size * gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if local_rank != -1 else 1)
    # if max_steps > 0:
    #     t_total = max_steps
    #     num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
    if num_train_epochs > 0:
        t_total = total_examples // batch_size * num_train_epochs
    max_steps = t_total
    model.to(device)
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()  
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    checkpoint_last = os.path.join(output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))   
    if local_rank == 0:
        torch.distributed.barrier()

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank%gpu_per_node],
                                                          output_device=local_rank%gpu_per_node,
                                                          find_unused_parameters=True)

    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", total_examples )
    print("  Num epoch = %d", t_total*batch_size//total_examples)
    print("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
    print("  Total train batch size (w. parallel, distributed & accumulation) = %d", batch_size)
    print("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    print("  Total optimization steps = %d", t_total)
    
    global_step = 0
    tr_loss, logging_loss,avg_loss,tr_nb = 0.0, 0.0,0.0,0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    set_seed(seed)  # Added here for reproducibility (even between python 2 and 3)

    best_bleu = 0.0
 
    for idx in range(0, int(num_train_epochs)): 
        for step, (batch, token_labels) in enumerate(train_dataloader):
            inputs = batch.to(device)
            attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8, device=device)
            loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8, device=device)
            model.train()
            # outputs = model(inputs, attention_mask=attn_mask, labels=inputs, loss_mask=loss_mask)
            # loss = outputs[0]
            outputs = model(inputs, attention_mask=attn_mask)
            logits = outputs[0]
            labels = inputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
            ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
                
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % logging_steps == 0:
                    print("  steps: %s  ppl: %s", global_step, round(avg_loss,5))
                if local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / logging_steps, global_step)
                    logging_loss = tr_loss
                    tr_nb=global_step

                if local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save( os.path.join(output_dir, "training_bin"))
                    print("Saving model checkpoint to %s", output_dir)

                    # _rotate_checkpoints( checkpoint_prefix)
                    last_output_dir = os.path.join(output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save.save_pretrained(last_output_dir)
                    tokenizer.save_pretrained(last_output_dir)
                    idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                    with open(idx_file, 'w', encoding='utf-8') as idxf:
                        idxf.write(str(0) + '\n')

                    torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                    print("Saving optimizer and scheduler states to %s", last_output_dir)

                    step_file = os.path.join(last_output_dir, 'step_file.txt')
                    with open(step_file, 'w', encoding='utf-8') as stepf:
                        stepf.write(str(global_step) + '\n')

                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    # print("Saving optimizer and scheduler states to %s", output_dir)
                    

            if max_steps > 0 and global_step > max_steps:
                break
        if max_steps > 0 and global_step > max_steps:
            break

    if local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, tokenizer = get_gpt2()

    dataset = CodeGPTDataset(
        tokenizer=tokenizer,
        filepath="datasets/miniconcode/train.json",
        block_size=512,
        mode="train"
    )

    fh = None
    pool = None

    train(
        dataset,
        model,
        tokenizer,
        "../models/concode_testing",
        device=device,
        tensorboard_dir="../models/concode_testing/tensorboard"
    )