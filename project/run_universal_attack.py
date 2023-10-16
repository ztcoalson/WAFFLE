import os.path
import logging
import random
import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm, trange
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from datasets import load_dataset
from pabee.modeling_pabee_bert import BertForSequenceClassificationWithPabee
from pabee_model_wrapper import *
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassificationWithPabee, BertTokenizer),
}

# Turn dataset into model-usable form
def generate_dataset(examples, tokenizer, trigger=""):
    # Tokenize
    new_examples = [trigger + e for e in examples["sentence"]]
    features = tokenizer(new_examples, padding='max_length', max_length=128)
    labels = [round(l) for l in examples["label"]]

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([id for id in features.input_ids], dtype=torch.long)
    all_attention_mask = torch.tensor([a for a in features.attention_mask], dtype=torch.long)
    all_token_type_ids = torch.tensor([t_id for t_id in features.token_type_ids], dtype=torch.long)
    all_labels = torch.tensor([label for label in labels], dtype=torch.long)

    # Wrap with TensorDataset
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    
    return dataset

# ----------------------------------------
#   Compute the efficacy; AUC under the cumulative function
# ----------------------------------------
def efficacy(stats):
    if sum(stats) != 0:
        tot_probs = np.cumsum(stats) / sum(stats)
        tot_fracs = (np.arange(len(stats)) + 1) / len(stats)
        efficacy  = np.trapz(tot_probs, tot_fracs)
    else:
        efficacy  = 0.
    return efficacy

# Simple eval to report accuracy and efficacy
def evaluate(model, eval_dataset, patience=6):

    model.bert.set_regression_threshold(0)
    model.bert.set_patience(patience)
    model.bert.reset_stats()
    model.to("cuda")

    results = {}

    eval_output_dir = "./uni_attack_cache/pabee/eval/"

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", 1)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    total_num_layers = -1
    terminal_inf_layer = list()
    exit_results = [0 for _ in range(13)]
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to("cuda") for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }

            outputs, total_num_layers_, terminal_inf_layer_, _ = model(**inputs)

            total_num_layers_ = total_num_layers_[0].item()
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
            exit_results[terminal_inf_layer_[0].item()] += 1

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            terminal_inf_layer = terminal_inf_layer_.detach().cpu().numpy()

        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            terminal_inf_layer = np.append(terminal_inf_layer, terminal_inf_layer_.detach().cpu().numpy(), axis=0)

    total_num_layers=max(total_num_layers_,total_num_layers)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    matches = (preds == out_label_ids).astype(int)

    results = {"acc": sum(matches) / len(matches)}
    results["eff"] = efficacy(exit_results)

    model.bert.log_stats()

    return results

# Train model (optimized for sst2)
def train(model, train_dataset, model_path, epochs):
    """ Train the model """
    train_dataloader = DataLoader(train_dataset, batch_size=16)

    t_total = len(train_dataloader) * epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(model_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(model_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(model_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", epochs)
    logger.info("  Instantaneous batch size per GPU = %d", 128)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d", 128)
    logger.info("  Gradient Accumulation steps = %d", 1)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        epochs,
        desc="Epoch",
    )
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for _, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to("cuda") for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
            }

            outputs,_,_,_ = model(**inputs)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

    return global_step, tr_loss / global_step

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_slowdown(example, model, tokenizer, v):
    # Tokenize input and prepare for model
    perturbed_tokens = tokenizer(v + " " + example["sentence"], padding="max_length", max_length=128)
    inputs = {
        "input_ids": torch.tensor([perturbed_tokens.input_ids], dtype=torch.long).to("cuda"),
        "token_type_ids": torch.tensor([perturbed_tokens.token_type_ids], dtype=torch.long).to("cuda"),
        "attention_mask": torch.tensor([perturbed_tokens.attention_mask], dtype=torch.long).to("cuda"),
        "labels": torch.tensor([round(example["label"])], dtype=torch.long).to("cuda")
    }
    
    # Compute slowdown (our objective function)
    with torch.no_grad():
        model.to("cuda")
        model.bert.set_regression_threshold(0)
        model.bert.set_patience(6)
        model.bert.reset_stats()
        model.eval()
        _, _, _, all_logits = model(**inputs)
    
    softmax = torch.nn.Softmax(dim=1)

    all_logits = [softmax(a).cpu() for a in all_logits]

    tot_score  = 0.

    # uniform tensor
    nclasses = 2
    eachprob = 1. / nclasses
    uni_prob = [eachprob for _ in range(nclasses)]
    uni_prob = torch.Tensor(uni_prob)

    # loop over the layer outputs
    for layer_output in all_logits:
        # : this will bound the loss in [0, 1]
        tot_score += (1. - (1. / (nclasses - 1)) * sum(abs(layer_output - uni_prob).squeeze())).item()

    # convert the tensor to a float value
    return tot_score / len(all_logits)

# Perform random attack
def random_attack(model, targeted_data, vocab, tokenizer, seed=42, p=1, k=3):
    # Shuffle and cut dataset
    targeted_data.shuffle(seed=seed)
    num_samples = round(len(targeted_data["sentence"]) * p)
    cut_data = [targeted_data[i] for i in range(num_samples)]

    # Compute "importance" of all v in vocab
    best = {"word": "", "slowdown": float("-inf")}
    for v in tqdm(vocab, desc="Vocab Seen"):
        slowdown = 0
        for example in tqdm(cut_data, total=num_samples, desc="Attacking"):
            slowdown += compute_slowdown(example, model, tokenizer, v)
        if slowdown > best["slowdown"]:
            best = {"word": v, "slowdown": slowdown}
    
    # Choose top k most important words to build trigger sequence
    trigger = ""
    for _ in range(k):
        trigger += best["word"] + " "
    
    # Compute slowdown of trigger sequence
    print(evaluate(model, generate_dataset(targeted_data, tokenizer), patience=6))
    print(evaluate(model, generate_dataset(targeted_data, tokenizer, trigger), patience=6))

def main():
    seed = 42

    set_seed(seed)

    # Initialize model, cuda(), and optimizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]
    config = config_class.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        finetuning_task="sst2",
        cache_dir=None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True,
        cache_dir=None,
    )
    model = model_class.from_pretrained(
        "bert-base-uncased",
        from_tf=False,
        config=config,
        cache_dir=None,
    )
    model.to("cuda")

    # where to save the model
    model_path = f"./uni_attack_cache/pabee/"

    # dataset and vocab
    sst2 = load_dataset("sst2", "default")
    vocab = []
    with open(f'./uni_attack_cache/pabee/vocab.txt') as vocab_file:
        for line in vocab_file:
            if "#" not in line and "[" not in line:
                vocab.append(line.strip())
    random.shuffle(vocab)
    

    # if the model already exists (its been trained), load the pre-trained weights and vocabulary
    if os.path.isfile(model_path + "pytorch_model.bin"):
        model = model_class.from_pretrained(model_path)
    # otherwise train model from scratch and save its weights
    else:
        train(model, generate_dataset(sst2["train"], tokenizer), model_path, 3)
        model.save_pretrained(model_path)
    
    # launch random attack
    random_attack(model, sst2["validation"], vocab[:1000], tokenizer, seed=seed, p=0.10, k=1)

if __name__ == '__main__':
    main()
