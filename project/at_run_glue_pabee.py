"""
    To run the adversarial training of a model, adapted from:
    https://github.com/QData/TextAttack-A2T
    (Note: no multi-GPU support)
"""
import os, sys
import math
import glob
import json
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter, OrderedDict
try:
    from project.pabee_model_wrapper import PabeeModelWrapper
except:
    sys.path.append('..')
    from project.pabee_model_wrapper import PabeeModelWrapper

# torch ...
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from pabee.modeling_pabee_albert import AlbertForSequenceClassificationWithPabee
from pabee.modeling_pabee_bert import BertForSequenceClassificationWithPabee

# transformers
import transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from datasets import load_dataset
from transformers import InputExample

# for adversarial training
import textattack
from textattack.attack_results import MaximizedAttackResult, SuccessfulAttackResult, FailedAttackResult
from textattack.constraints.pre_transformation import InputColumnModification

# custom...
from utils import _get_column_names


# make the script quite
transformers.logging.set_verbosity_error()

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# disable warning (too much)
import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassificationWithPabee, BertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassificationWithPabee, AlbertTokenizer),
}


# ----------------------------------------
#   Misc. functions
# ----------------------------------------
def _set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _load_datasets(taskname, data_dir, glue_script):
    # number of labels / output mode
    processor = processors[taskname]()
    out_mode  = output_modes[taskname]
    labellist = processor.get_labels()
    num_label = len(labellist)

    # rename the dataset
    # (some dataset names are inconsistent over the library APIs)
    if taskname == 'sst-2':
        taskname = 'sst2'

    # load trainset
    train_file = os.path.join(data_dir, 'train.tsv')
    train_data = load_dataset(
            glue_script, 
            taskname, 
            data_files=train_file, 
            cache_dir=None, 
            split='train')
    train_in, train_out = _get_column_names(train_data[0], "train", processor)
    train_data = textattack.datasets.HuggingFaceDataset(train_data, dataset_columns=(train_in, train_out))

    # load validset
    # (Note: we work on the dev dataset, not the validation dataset)
    if 'mnli' in taskname:
        valid_file  = os.path.join(data_dir, 'dev_mismatched.tsv')
        valid_split = 'validation_matched'
    else:
        valid_file = os.path.join(data_dir, 'dev.tsv')
        valid_split = 'validation'
    valid_data = load_dataset(
            glue_script, 
            taskname, 
            data_files=valid_file, 
            cache_dir=None, 
            split=valid_split)
    valid_in, valid_out = _get_column_names(valid_data[0], "dev", processor)
    valid_data = textattack.datasets.HuggingFaceDataset(valid_data, dataset_columns=(valid_in, valid_out))

    # return ...    
    return train_data, valid_data, out_mode, num_label


def _control_multiexit(args, model, enable=False):
    if args.model_type == 'albert':
        model.albert.multiexit = enable
    elif args.model_type == 'bert':
        model.bert.multiexit = enable
    else:
        assert NotImplementedError()
    # dopne.


def _craft_adv_examples(args, dataset, model, attack, entire=False):
    # init.
    model.eval()

    # temporarily enable the multi-exit
    _control_multiexit(args, model, enable=True)

    # set the numbers
    if not entire:
        num_train_adv_examples = math.ceil(len(dataset) * args.num_adv_examples)
    else:
        num_train_adv_examples = len(dataset)
    print (' : craft {} over {} adv. examples with {}'.format( \
        num_train_adv_examples, len(dataset), attack.__class__.__name__))

    # set the attack arguments
    attack_args = textattack.AttackArgs(
        num_examples=num_train_adv_examples,
        num_examples_offset=0,
        query_budget=args.attack_maxquery,
        shuffle=True,
        disable_stdout=True,
        silent=True,
        log_to_csv=None,
    )

    # run attacks
    attacker = textattack.Attacker(attack, dataset, attack_args)
    results  = attacker.attack_dataset()

    # disable multi-exit
    _control_multiexit(args, model, enable=False)

    # log the results
    attacker
    attack_types  = Counter(r.__class__.__name__ for r in results)
    total_attacks = (attack_types["SuccessfulAttackResult"] + attack_types["FailedAttackResult"])
    success_rate  = attack_types["SuccessfulAttackResult"] / total_attacks * 100
    logger.info(f"Attack {num_train_adv_examples} samples")
    logger.info(f" : Total attack results: {len(results)}")
    logger.info(f" : Success: {success_rate:.2f}% [{attack_types['SuccessfulAttackResult']} / {total_attacks}]")

    # compose a loader for training
    if args.attack_goal == 'ours':
        adv_examples = [
            (
                tuple(r.perturbed_result.attacked_text._text_input.values())
                + ("adversarial_example",),
                r.perturbed_result.ground_truth_output,
            )
            for r in results
            if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult, FailedAttackResult))
        ]

    elif args.attack_goal == 'base':
        adv_examples = [
            (
                tuple(r.perturbed_result.attacked_text._text_input.values())
                + ("adversarial_example",),
                r.perturbed_result.ground_truth_output,
            )
            for r in results
            if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult))
        ]

    else:
        assert False, ('Error: unsupported attack goal - {}, abort'.format(args.attack_goal))

    adv_dataset = textattack.datasets.Dataset(
        adv_examples,
        input_columns=dataset.input_columns + ["_example_type",],
        # input_columns=dataset.input_columns,
        label_map=dataset.label_map,
        label_names=dataset.label_names,
        output_scale_factor=dataset.output_scale_factor,
        shuffle=False,
    )
    return adv_dataset


def _set_input_modification(args):
    # 1st column
    if args.task_name == "rte" or args.task_name == "wnli" or \
        args.task_name == "mnli" or args.task_name == "mnli-mm":
        return InputColumnModification(["sentence1", "sentence2"], {"sentence2"})

    # 2nd column
    elif args.task_name == "qnli":
        return InputColumnModification(["sentence1", "sentence2"], {"sentence1"})

    # one or the other (ideally though whichever one is larger)
    elif args.task_name == "stsb" or args.task_name == "qqp" or \
        args.task_name == "mrpc":
        return InputColumnModification(["sentence1", "sentence2"], {"sentence2"})

    # has one column, modify any columns, other cases (cola, sst2)
    else:
        return InputColumnModification([], {})

    # done.


def _compose_dataloader(clean_dataset, adver_dataset, batch_size=1, valid=False):
    # function that's used by dataloader
    def collate_fn(data):
        input_texts = []
        targets = []
        is_adv_sample = []
        for item in data:
            if "_example_type" in item[0].keys():

                # Get example type value from OrderedDict and remove it
                adv = item[0].pop("_example_type")

                # with _example_type removed from item[0] OrderedDict
                # all other keys should be part of input
                _input, label = item
                if adv != "adversarial_example":
                    raise ValueError(
                        "`item` has length of 3 but last element is not for marking if the item is an `adversarial example`."
                    )
                else:
                    is_adv_sample.append(True)
            else:
                # else `len(item)` is 2.
                _input, label = item
                is_adv_sample.append(False)

            if isinstance(_input, OrderedDict):
                _input = tuple(_input.values())
            else:
                _input = tuple(_input)

            if len(_input) == 1:
                _input = _input[0]
            input_texts.append(_input)
            targets.append(label)

        return input_texts, torch.tensor(targets), torch.tensor(is_adv_sample)


    # add the adversarial examples
    if adver_dataset:
        train_dataset = torch.utils.data.ConcatDataset([clean_dataset, adver_dataset])
    else:
        train_dataset = clean_dataset

    # compose
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False if valid else True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader


# ----------------------------------------
#   Train/valid functions
# ----------------------------------------
def train(args, train_dataset, model, tokenizer, output_mode):
    # prep.
    tb_writer = SummaryWriter()

    # set the seed
    _set_seed(args)


    #  compute the total clean training steps
    tot_training_steps = math.ceil(
        len(train_dataset) / (args.train_batch_size * args.gradient_accumulation_steps)
    ) * args.num_clean_epochs


    # ----------------------------------------
    #   Attacks used for adversarial training
    # ----------------------------------------
    if args.adv_train:
        # : wrap the one
        model_wrapper = PabeeModelWrapper(
            model, tokenizer, \
            args.max_seq_length,
            True,                   # set the multiexit to True
            args.attack_goal,
        )

        # : set the attack
        attack = textattack.attack_recipes.A2TYoo2021.build(
            model_wrapper, _set_input_modification(args), args)

        # : additional configuration for the multi-exit models
        attack.goal_function.batch_size = 1
        attack.goal_function.regression_threshold = 0.5
        attack.goal_function.use_cache = True

        # : # of training steps
        tot_adver_training_steps = math.ceil(
            (len(train_dataset) + math.ceil(len(train_dataset) * args.num_adv_examples))
            / (args.train_batch_size * args.gradient_accumulation_steps)
        ) * (args.num_train_epochs - args.num_clean_epochs)

        # : add to the total
        tot_training_steps += tot_adver_training_steps


    # set the loss we will use
    # (Note: set reduction to 'none' if we want to weigh adv. example differently)
    if output_mode == "regression":
        loss_fn = nn.MSELoss(reduction="mean")
    else:
        loss_fn = nn.CrossEntropyLoss(reduction="mean")


    # set the optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup( \
        optimizer, \
        num_warmup_steps=args.warmup_steps, \
        num_training_steps=tot_training_steps)

    # check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) \
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # : load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))


    # write out the configurations
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", tot_training_steps)


    # data holaders
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # : set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])

        # : compute so far
        epochs_trained = global_step // \
            (len(train_dataset) \
                // args.train_batch_size \
                // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % \
            (len(train_dataset) \
                // args.train_batch_size \
                // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    # losses holders
    tr_loss, logging_loss = 0.0, 0.0


    # ----------------------------------------
    #   Run training
    # ----------------------------------------
    # train_iterator = trange(
    #     epochs_trained,
    #     int(args.num_train_epochs),
    #     desc="Epoch",
    #     disable=False,
    # )

    # loop over the epochs
    for epoch in range(args.num_train_epochs):

        # : craft adversarial examples
        if args.adv_train and (epoch >= args.num_clean_epochs):
        # if args.adv_train:
            adv_dataset = _craft_adv_examples(args, train_dataset, model, attack)
        else:
            adv_dataset = None

        # : compoase a dataloader, mix clean and adv examples
        train_loader = _compose_dataloader( \
            train_dataset, adv_dataset, batch_size=args.train_batch_size)

        # : run over the loader
        epoch_iterator = tqdm(train_loader, desc=" Adv-train:{}".format(epoch))
        for step, batch in enumerate(epoch_iterator):

            # : skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            # : set model to train
            model.train()

            # : compose a batch
            input_texts, targets, is_adv_sample = batch
            _targets = targets
            targets = targets.to(args.device)

            # : tokenize the batch
            if isinstance(model, transformers.PreTrainedModel) \
                or isinstance(model.module, transformers.PreTrainedModel):
                input_ids = tokenizer(
                    input_texts,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )
                input_ids.to(args.device)
                logits, _, _, _ = model(**input_ids)
                logits = logits[0]          # model outputs are always tuple in transformers

            else:
                input_ids = tokenizer(input_texts)
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids)
                input_ids = input_ids.to(args.device)
                logits, _, _, _ = model(input_ids)
                logits = logits[0]

            # : compute the loss
            loss = loss_fn(logits.squeeze(), targets.squeeze())
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # : compute the gradients
            loss.backward()

            # : record the stats.
            tr_loss += loss.item()

            # : etc...
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Note: update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # : end for step, batch...

    # close the tensorboard
    tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(args, valid_dataset, model, tokenizer, output_mode, prefix="", patience=0):
    # set the model to eval mode
    if args.model_type == "albert":
        model.albert.set_regression_threshold(args.regression_threshold)
        model.albert.set_patience(int(args.patience))
        model.albert.reset_stats()
        model.albert.multiexit=args.multiexit
    elif args.model_type == "bert":
        model.bert.set_regression_threshold(args.regression_threshold)
        model.bert.set_patience(int(args.patience))
        model.bert.reset_stats()
        model.bert.multiexit=args.multiexit
    else:
        raise NotImplementedError()

    # create output dirs.
    valid_output_dir = args.output_dir if args.task_name != 'mnli' else args.output_dir + "_" + str(patience) + "-MM"
    if not os.path.exists(valid_output_dir): os.makedirs(valid_output_dir)

    # create dataloader
    valid_loader = _compose_dataloader(valid_dataset, None, batch_size=args.valid_batch_size, valid=True)

    # ----------------------------------------
    #   Attacks used for testing the robustness
    # ----------------------------------------
    if args.adv_train:
        # : wrap the one
        model_wrapper = PabeeModelWrapper(
            model, tokenizer, \
            args.max_seq_length,
            True,                   # set the multiexit to True
            args.attack_goal,
        )

        # : set the attack
        attack = textattack.attack_recipes.A2TYoo2021.build(
            model_wrapper, _set_input_modification(args), args)

        # : additional configuration for the multi-exit models
        attack.goal_function.batch_size = 1
        attack.goal_function.regression_threshold = 0.5
        attack.goal_function.use_cache = True

        # : run crafting
        adver_dataset = _craft_adv_examples(args, valid_dataset, model, attack, entire=True)

        # : compose the adversarial example loader
        adver_loader = _compose_dataloader(adver_dataset, None, batch_size=args.valid_batch_size, valid=True)


    # set the loss we will use
    # (Note: set reduction to 'none' if we want to weigh adv. example differently)
    if output_mode == "regression":
        loss_fn = nn.MSELoss(reduction="mean")
    else:
        loss_fn = nn.CrossEntropyLoss(reduction="mean")

    # run...
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  # examples: %d", len(valid_dataset))
    logger.info("  Batch size: %d", args.valid_batch_size)

    # data-holders
    results = {}


    # run validation with the clean data
    vloss, vindexes, vpredicts, vtargets, vexitnums, vmatches, tot_numexits = \
        _evaluate(args, valid_loader, model, tokenizer, loss_fn, output_mode)

    # compute the metrics
    vresult = compute_metrics(args.task_name, vpredicts, vtargets)
    tresult = {}
    for key, value in vresult.items():
        if key == 'acc': key = 'clean-acc'
        tresult[key] = value
    results.update(tresult)

    # write out
    strng = "matches_p" + str(patience)+".clean.txt"
    fpath =  os.path.join(valid_output_dir, prefix, strng)
    np.savetxt(fpath, vmatches, fmt='%i')

    strng = "exit_layer_p" + str(patience)+".clean.txt"
    fpath =  os.path.join(valid_output_dir, prefix, strng)
    np.savetxt(fpath, vexitnums, fmt='%i')

    strng = "idx_p" + str(patience)+".clean.txt"
    fpath =  os.path.join(valid_output_dir, prefix, strng)
    np.savetxt(fpath, vindexes, fmt='%i')


    # ----------------------------------------
    #   Evaluate the robustness
    # ----------------------------------------
    if args.adv_train:

        # : run validation with the adversarial examples
        aloss, aindexes, apredicts, atargets, aexitnums, amatches, _ = \
            _evaluate(args, adver_loader, model, tokenizer, loss_fn, output_mode)

        # : compute the metrics
        aresult = compute_metrics(args.task_name, apredicts, atargets)
        tresult = {}
        for key, value in aresult.items():
            if key == 'acc': key = 'adver-acc'
            tresult[key] = value
        results.update(tresult)

        # : write out
        strng = "matches_p" + str(patience)+".adver.txt"
        fpath =  os.path.join(valid_output_dir, prefix, strng)
        np.savetxt(fpath, amatches, fmt='%i')

        strng = "exit_layer_p" + str(patience)+".adver.txt"
        fpath =  os.path.join(valid_output_dir, prefix, strng)
        np.savetxt(fpath, aexitnums, fmt='%i')

        strng = "idx_p" + str(patience)+".adver.txt"
        fpath =  os.path.join(valid_output_dir, prefix, strng)
        np.savetxt(fpath, aindexes, fmt='%i')


    # write out the total results
    strng = "eval_L" + str(tot_numexits) + "_p" + str(patience) + "_results.txt"
    valid_output_file = os.path.join(valid_output_dir, prefix, strng)
    with open(valid_output_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            # print("  %s = %s" % (key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(results[key])))

    # if args.eval_all_checkpoints and patience != 0:
    if patience != 0:
        if args.model_type == "albert":
            model.albert.log_stats()
        elif args.model_type == "bert":
            model.bert.log_stats()
        else:
            raise NotImplementedError()

    return results



def _evaluate(args, dataloder, model, tokenizer, loss_fn, output_mode):

    # data-holders (metrics...)
    valid_loss   = 0.0
    all_indexes  = []
    all_predicts = []
    all_targets  = []
    all_exitnums = []
    tot_numexits = -1

    # loop over the dataloder
    for step, batch in tqdm(enumerate(dataloder), desc=" Evaluate"):
        # : set model to eval.
        model.eval()

        # : compose a batch
        input_texts, targets, _ = batch
        _targets = targets
        targets = targets.to(args.device)

        # : run forward
        with torch.no_grad():
            input_ids = tokenizer(
                input_texts,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )
            input_ids.to(args.device)
            
            # :: outputs
            outputs, final_num, exit_num, logits = model(**input_ids)

            # :: formatting
            exit_num = exit_num.item()

            # :: compute the loss
            if tot_numexits < 0: tot_numexits = len(logits)
            exit_logit = logits[exit_num-1]                             # exit_num is in [1, 12 (end)]
            valid_loss += loss_fn(exit_logit, targets).item()

        # : compute the metrics
        if output_mode == "regression":
            prediction = exit_logit[0]
        else:
            prediction = exit_logit[0].argmax(dim=0)

        all_indexes.append(step)
        all_predicts.append(prediction.item())
        all_targets.append(_targets.item())
        all_exitnums.append(exit_num)

    # end for ...

    # formatting...
    valid_loss /= len(dataloder)
    all_indexes  = np.array(all_indexes)
    all_predicts = np.array(all_predicts)
    all_targets  = np.array(all_targets)
    all_exitnums = np.array(all_exitnums)

    # compute the matching...
    if output_mode == "regression":
        all_matches = []
        for pidx, each_predict in enumerate(all_predicts):
            each_target = all_targets[pidx]
            if each_target:
                each_plabel = (each_predict >= each_target - args.regression_threshold)
            else:
                each_plabel = (each_predict <= each_target + args.regression_threshold)
            all_matches.append((each_plabel == each_target))
        all_matches = np.array(all_matches).astype(int)
    else:
        all_matches = (all_predicts == all_targets).astype(int)

    # return items
    return valid_loss, all_indexes, all_predicts, all_targets, all_exitnums, all_matches, tot_numexits


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name.",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--hub_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where cached features are stored when data is downloaded \
            from huggingface hub.",
    )
    parser.add_argument(
        "--patience",
        default="0",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--regression_threshold",
        default=0,
        type=float,
        required=False,
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--glue_script",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--valid_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        type=int, default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    # ----------------------------------------
    #   To run the adversarial training
    # ----------------------------------------
    parser.add_argument(
        "--adv_train",
        action="store_true",
        help="enable adversarial training")
    parser.add_argument(
        "--num_clean_epochs",
        type=int, default=1,
        help="number of epochs for clean training (default: 1)")
    parser.add_argument(
        "--num_adv_examples",
        type=float, default=0.2,
        help="percentage of the adversarial examples (default: 0.2)")
    parser.add_argument(
        "--attack_goal",
        type=str, default='base',
        help="base / oavg / ours")
    parser.add_argument(
        "--attack_maxquery",
        type=int, default=1000,
        help="max. number of queries per attack sample (default: 1000)")
    parser.add_argument(
        "--attack_threshold",
        type=float, default=1.0,
        help="slowdown score; not used for the base and oavg (default: 1.0)")
    parser.add_argument(
        "--multiexit",
        action="store_true",
        help="enable multi-exit")

    args = parser.parse_args()


    # ----------------------------------------
    #   Initialization
    # ----------------------------------------
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # setup CUDA and GPUs
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    logger.warning(
        "Device: %s, n_gpu: %s, no distributed training.",
        device, args.n_gpu,
    )

    # set seed
    _set_seed(args)
    

    # ----------------------------------------
    #   Datasets
    # ----------------------------------------

    # task name correction
    args.task_name = args.task_name.lower()
    if args.task_name == 'mnli-mm':
        args.task_name = 'mnli_mismatched'
    elif args.task_name == "sts-b":
        args.task_name = "stsb"

    # train/test dataload
    train_dataset, valid_dataset, output_mode, num_labels = \
        _load_datasets(args.task_name, args.data_dir, args.glue_script)


    # sanity checks: adversarial training
    if args.patience != "0" and args.valid_batch_size != 1:
        raise ValueError("The eval batch size must be 1 with PABEE inference on.")


    # load the model
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # set model to GPUs
    model.to(args.device)


    # print out model stats
    print ('Model stats:')
    print (' - # parameters : ', sum(param.numel() for param in model.parameters()))
    output_layers_param_num = sum(param.numel() for param in model.classifiers.parameters())
    print (' - # param (out): ', output_layers_param_num)
    single_output_layer_param_num = sum(param.numel() for param in model.classifiers[0].parameters())
    print (' - # param (add): ', output_layers_param_num - single_output_layer_param_num)


    # check before running
    logger.info("Training/evaluation parameters %s", args)


    # training
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, output_mode)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # saving best-practices: 
    # Use defaults names for the model, to reload it by using from_pretrained()
    if args.do_train:
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)


    # run evaluation
    results = {}
    if args.do_eval:
        patience_list = [int(x) for x in args.patience.split(",")]
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            print(f"Evaluation for checkpoint {prefix}")
            for patience in patience_list:
                result = evaluate(args, valid_dataset, model, tokenizer, output_mode, prefix=prefix, patience=patience)
                result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                results.update(result)
    return results


if __name__ == "__main__":
    main()
