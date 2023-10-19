"""
    A script for attacking multi-exit models
"""
# basics
import os, re
import json
import argparse
import numpy as np
from tqdm import tqdm

# torch
from torch.utils.data import TensorDataset, DataLoader

# transformers
from transformers import InputExample
from transformers import glue_processors as processors

# custom...
from datasets import load_dataset
from utils import _get_column_names

# custom (models)...
from pabee.modeling_pabee_albert import AlbertForSequenceClassificationWithPabee
from pabee.modeling_pabee_bert import BertForSequenceClassificationWithPabee

from deebert.modeling_highway_bert import DeeBertForSequenceClassification

from pastfuture.modeling_albert import AlbertForSequenceClassification
from pastfuture.modeling_bert import BertForSequenceClassification

# configurations
from transformers import (
    BertConfig,
    BertTokenizer,
    AlbertConfig,
    AlbertTokenizer
)

# custom (other wrappers)...
from pabee_model_wrapper import *
from pastfuture_model_wrapper import *
from deebert_model_wrapper import *

# textattacks
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import BERT
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    MaxModificationRate,
    RepeatModification,
    StopwordModification,
)
from textattack.transformations import WordSwapEmbedding



# --------------------------------------------------------------------------------
#   Globals
# --------------------------------------------------------------------------------
MAX_LAYERS  = 12
ATTACK_FAIL = 0
ATTACK_SUCC = 1
ATTACK_SKIP = 2
ATTACK_ALLS = 3

# use cuda if available
_use_cuda = torch.cuda.is_available()


# --------------------------------------------------------------------------------
#   Misc. function
# --------------------------------------------------------------------------------
_outfile = None
def set_outfile(ofile):
    global _outfile
    _outfile = ofile
    # done.

def print_(x):
    global _outfile
    if not _outfile: print (x)
    else:            print (x, file=_outfile)
    # done.

def _compose_dataloader(dataset, tokenizer, batch_size=1):
    # tokenize
    examples = [e for e in dataset["sentence"]]
    features = tokenizer(examples, padding='max_length', max_length=128)
    labels   = [round(l) for l in dataset["label"]]

    # convert to tensor and build dataset
    all_input_ids      = torch.tensor([id for id in features.input_ids], dtype=torch.long)
    all_attention_mask = torch.tensor([a for a in features.attention_mask], dtype=torch.long)
    all_token_type_ids = torch.tensor([t_id for t_id in features.token_type_ids], dtype=torch.long)
    all_labels         = torch.tensor([label for label in labels], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    # compose a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


# --------------------------------------------------------------------------------
#   Evaluation funcntion
# --------------------------------------------------------------------------------
def evaluate(args, model, dataset, tokenizer):
    # compose the loader
    dataloader = _compose_dataloader(dataset, tokenizer, batch_size=1)

    # set the stats to zero
    if args.model_type == "albert":
        model.albert.set_regression_threshold(args.regression_threshold)
        model.albert.set_patience(int(args.patience))
        model.albert.reset_stats()
    elif args.model_type == "bert":
        model.bert.set_regression_threshold(args.regression_threshold)
        model.bert.set_patience(int(args.patience))
        model.bert.reset_stats()
    else:
        raise NotImplementedError()

    # move model to GPU
    if _use_cuda: model.to("cuda")

    # run evals
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    total_num_layers = -1
    terminal_inf_layer = list()
    exit_results = [0 for _ in range(13)]

    # loop over the entire data
    for batch in tqdm(dataloader, desc=" : [run-eval]"):
        model.eval()
        if _use_cuda:
            batch = tuple(t.to("cuda") for t in batch)
        else:
            batch = tuple(t for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids"     : batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels"        : batch[3]
            }

            outputs, total_num_layers_, terminal_inf_layer_, all_logits = model(**inputs)

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

    total_num_layers = max(total_num_layers_,total_num_layers)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    matches = (preds == out_label_ids).astype(int)

    # report the results
    model.albert.log_stats()
    print ('*** Acc. [{}] and Eff. [{}] ***'.format( \
        sum(matches) / len(matches), efficacy(exit_results)))
    
    # done.


# --------------------------------------------------------------------------------
#   Attack function
# --------------------------------------------------------------------------------
def _compute_slowdown(flogit, nexits, tlogits):
    loss = 0.

    # compose uniform probability
    nclasses = len(flogit[-1][0])
    eachprob = 1. / nclasses
    uni_prob = [eachprob for _ in range(nclasses)]
    uni_prob = torch.Tensor(uni_prob)
    uni_prob = uni_prob.to("cuda") if _use_cuda else \
               uni_prob

    # compute the loss (between 0 and 1)
    for each_logit in tlogits:
        loss += (1. - (1. / (nclasses - 1)) * sum(abs(F.softmax(each_logit, dim=1)[0] - uni_prob)))
    loss /= (nexits.item() + 1)
    return loss

def _compose_input(sample, tokenizer):
    """
        Sample's format:
        {'sentence': "it 's a charming and often affecting journey . ", 'label': 1, 'idx': 0}
    """
    feature = tokenizer(sample['sentence'], padding='max_length', max_length=128)
    label   = round(sample['label'])

    # compose the model input
    each_input = {
        "input_ids"     : torch.tensor([feature.input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([feature.attention_mask], dtype=torch.long),
        "token_type_ids": torch.tensor([feature.token_type_ids], dtype=torch.long),
        "labels"        : torch.tensor([label], dtype=torch.long),
    }

    # set to cuda
    if _use_cuda:
        each_input = { k: v.to('cuda') for k, v in each_input.items() }
    return each_input

def _compute_w2t_mapping(sample, model_wrapper):
    """
        Text align...
    """
    # convert them into words and tokens
    words  = _words_from_text(sample['sentence'])
    tokens = model_wrapper.tokenize([sample['sentence']], strip_prefix=True)[0]

    # test
    constraints = [RepeatModification(), StopwordModification()]
    constraints.append(InputColumnModification([], {}))
    constraints.append(PartOfSpeech(allow_verb_noun_swap=False))
    constraints.append(MaxModificationRate(max_rate=0.1, min_threshold=4))
    sent_encoder = BERT(
        model_name="stsb-distilbert-base", threshold=0.9, metric="cosine"
    )
    constraints.append(sent_encoder)
    transformation = WordSwapEmbedding(max_candidates=20)
    indices_to_order = transformation(sample['sentence'], transformation, return_indices=True)
    print (indices_to_order)


    print (words)
    print (tokens)

    # data holders
    word2token_mapping = {}
    j = 0
    last_matched = 0

    # loop over the data
    for i, word in enumerate(words):
        matched_tokens = []
        while j < len(tokens) and len(word) > 0:
            token = tokens[j].lower()
            idx = word.lower().find(token)
            if idx == 0:
                word = word[idx + len(token) :]
                matched_tokens.append(j)
                last_matched = j
            j += 1

        if not matched_tokens:
            word2token_mapping[i] = None
            j = last_matched
        else:
            word2token_mapping[i] = matched_tokens

    print (word2token_mapping)
    exit()

    return word2token_mapping

def _words_from_text(s, words_to_ignore=[]):
    """
        Lowercases a string, removes all non-alphanumeric characters, and splits into words.
    """
    s = " ".join(s.split())

    homos = """˗৭Ȣ𝟕бƼᏎƷᒿlO`ɑЬϲԁе𝚏ɡհіϳ𝒌ⅼｍոорԛⲅѕ𝚝սѵԝ×уᴢ"""
    exceptions = """'-_*@"""
    filter_pattern = homos + """'\\-_\\*@"""
    # TODO: consider whether one should add "." to `exceptions` (and "\." to `filter_pattern`)
    # example "My email address is xxx@yyy.com"
    filter_pattern = f"[\\w{filter_pattern}]+"
    words = []
    for word in s.split():
        # Allow apostrophes, hyphens, underscores, asterisks and at signs as long as they don't begin the word.
        word = word.lstrip(exceptions)
        filt = [w.lstrip(exceptions) for w in re.findall(filter_pattern, word)]
        words.extend(filt)
    words = list(filter(lambda w: w not in words_to_ignore + [""], words))
    return words


def run_attack(args, dataset, model, model_wrapper, tokenizer, niter=1, tword="the"):
    # ----------------------------------------
    #   Prep. the model
    # ----------------------------------------
    # set the stats to zero
    if args.model_type == "albert":
        model.albert.set_regression_threshold(args.regression_threshold)
        model.albert.set_patience(int(args.patience))
        model.albert.reset_stats()
    elif args.model_type == "bert":
        model.bert.set_regression_threshold(args.regression_threshold)
        model.bert.set_patience(int(args.patience))
        model.bert.reset_stats()
    else:
        raise NotImplementedError()

    # move model to GPU
    if _use_cuda: model.to("cuda")

    # set the eval mode
    model.eval()

    # to compute the gradients
    embedding_layer = model.get_input_embeddings()
    embedding_state = embedding_layer.weight.requires_grad
    embedding_layer.weight.requires_grad = True

    embedding_grads = []
    def grad_hook(module, grad_in, grad_out):
        embedding_grads.append(grad_out[0])
    embedding_hooks = embedding_layer.register_backward_hook(grad_hook)


    # init. the trigger words
    trigger_words = ' '.join([tword] * args.attack_num_words)


    # ----------------------------------------
    #   Run attack...
    # ----------------------------------------
    for _ in range(niter):
        for each_data in tqdm(dataset, desc=' : [attack-{}]'.format(_)):
            # : append the trigger words
            each_data["sentence"] = ' '.join([trigger_words, each_data["sentence"]])
            
            # : compose the input
            each_input = _compose_input(each_data, tokenizer)

            # : clear out the gradients
            model.zero_grad()

            # : run forward
            final_logits, _, early_exits, total_logits = model(**each_input)

            # : compute the slowdown loss and run backward
            loss_slowdown = _compute_slowdown(final_logits, early_exits, total_logits)
            loss_slowdown.backward()

            # : compute the gradients
            if embedding_grads[0].shape[1] == 1:
                grad = torch.transpose(embedding_grads[0], 0, 1)[0].cpu().numpy()
            else:
                # if gradient has shape [1,max_sequence,_]
                grad = embedding_grads[0][0].cpu().numpy()
            
            # : do something with the gradients
            word2token_mapping = _compute_w2t_mapping(each_data, model_wrapper)
            print (word2token_mapping)
            exit()

            print (grad.shape)
            exit()


    # set the embedding layer back to the original state
    embedding_layer.weight.requires_grad = embedding_state
    embedding_hooks.remove()

    return None


def run_attacks(args):

    # compose the output string
    if 'ours' in args.attack_goal:
        attack_string = \
            "{}_{}_{}_{}_{}_t{}".format(
                args.attack, 
                'multi' if args.multiexit else 'single', 
                args.attack_goal, 
                args.attack_method, 
                args.attack_maxquery,
                args.attack_threshold
            )
    else:
        attack_string = \
            "{}_{}_{}_{}_{}".format(
                args.attack, 
                'multi' if args.multiexit else 'single', 
                args.attack_goal, 
                args.attack_method, 
                args.attack_maxquery
            )


    # load models (PABEE)
    if args.exit_type == 'pabee':
        basename = os.path.join( \
            args.result_dir, \
            'p{}_{}'.format(args.patience, attack_string))
        model_classes = {
            "bert": (BertConfig, BertForSequenceClassificationWithPabee, BertTokenizer),
            "albert": (AlbertConfig, AlbertForSequenceClassificationWithPabee, AlbertTokenizer),}
        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = model_classes[args.model_type]
        model = model_class.from_pretrained(args.model_path)

        if args.model_type == "albert":
            model.albert.set_regression_threshold(args.regression_threshold)
            model.albert.set_patience(int(args.patience))
            model.albert.reset_stats()
            tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=True)
            model.albert.multiexit=args.multiexit
        elif args.model_type == "bert":
            model.bert.set_regression_threshold(args.regression_threshold)
            model.bert.set_patience(int(args.patience))
            model.bert.reset_stats()
            tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=True)
            model.bert.multiexit=args.multiexit
        else:
            raise NotImplementedError()
        
        model_wrapper = PabeeModelWrapper( \
            model, tokenizer, \
            args.max_seq_length, \
            args.multiexit, \
            args.attack_goal)

    # load models (PASTFUTURE)
    elif args.exit_type == 'pastfuture':
        basename = os.path.join( \
            args.result_dir, \
            'e{}_{}'.format(args.entropy, attack_string))
        model_classes = {
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
            "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),}
        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = model_classes[args.model_type]
        model = model_class.from_pretrained(args.model_path)

        if args.model_type == "albert":
            model.albert.reset_stats()
            model.albert.set_patience(args.entropy)
            model.albert.multiexit=args.multiexit
            tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=True)
        elif args.model_type == "bert":
            model.bert.reset_stats()
            model.bert.set_patience(args.entropy)
            model.bert.multiexit=args.multiexit
            tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=True)
        else:
            raise NotImplementedError()

        model_wrapper = PastfutureModelWrapper( \
            model, tokenizer, \
            args.max_seq_length, \
            args.model_type, \
            args.multiexit, \
            args.attack_goal)
        
    # load models (DEEBERT)
    elif args.exit_type == 'deebert':
        basename = os.path.join( \
            args.result_dir, \
            'e{}_{}'.format(args.entropy, attack_string))
        model_classes = {
            "bert": (BertConfig, DeeBertForSequenceClassification, BertTokenizer),}
        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = model_classes[args.model_type]
        model = model_class.from_pretrained(args.model_path)

        if args.model_type == "bert":
            model.bert.encoder.set_early_exit_entropy(args.entropy)
            model.bert.encoder.multiexit=args.multiexit
            tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=True)
        else:
            raise NotImplementedError()
    
        model_wrapper = DeebertModelWrapper( \
            model, tokenizer, \
            args.max_seq_length, \
            args.multiexit, \
            args.attack_goal)

    # others (raise an error...)
    else:
        raise NotImplementedError()


    # task/data configuration
    _task  = args.task_name
    _data  = os.path.join(args.data_dir, 'dev.tsv')
    _split = 'validation'
    if args.task_name == 'mnli':
        _task = 'mnli_matched'
        _data = os.path.join(args.data_dir, 'dev_matched.tsv')
    elif args.task_name == 'mnli-mm':
        _task = 'mnli_mismatched'
        _data = os.path.join(args.data_dir, 'dev_mismatched.tsv')
    elif args.task_name == 'sst-2':
        _task = 'sst2'


    # load from the stored ones
    processor = processors[args.task_name]()
    n_samples = int(872 * 0.2)               # use 20% of samples to attack (sst-2)
    dataset_attk = load_dataset(
        args.glue_script, 
        _task, 
        data_files=_data, 
        cache_dir=None, 
        split="{}[:{}]".format(_split, n_samples))
    dataset_eval = load_dataset(
        args.glue_script, 
        _task, 
        data_files=_data, 
        cache_dir=None, 
        split=_split)
    
    # init. the triggers
    # dataset_attk = _compose_attack_datasets(dataset_attk, num_words=args.attack_num_words)


    # ----------------------------------------
    #   Run initial eval. (sanity-checks)
    # ----------------------------------------
    # evaluate(args, model, dataset_eval, tokenizer)      # default patience set to 6
    # evaluate(args, model, dataset_attk, tokenizer)
    # [Note] checked that the efficacy is the same for the eval and attk datasets


    # ----------------------------------------
    #   Run attack
    # ----------------------------------------
    run_attack(args, dataset_attk, model, model_wrapper, tokenizer)


    exit()


    # ----------------------------------------
    #   Store attack results (csv/txt)
    # ----------------------------------------
    # set the output files
    csv_output = basename + "_log.csv"
    txt_output = basename + "_log.txt"

    # set the logger
    log = textattack.loggers.attack_log_manager.AttackLogManager()
    log.add_output_csv(csv_output, None)
    log.add_output_file(txt_output, None)

    # store the results (use the slowdown metrics)
    log.log_results(results, slowdown=True)
    log.flush()


    # ----------------------------------------
    #   Store the results to a file in a custom format
    # ----------------------------------------
    # set the output files
    res_output = basename + "_results.txt"

    # store...
    outfile = open(res_output, "w")
    set_outfile(outfile)

    # failed attack case
    print_(">> Attack results (failed) <<")
    orig_exit_stats, pert_exit_stats = layer_info(results, ATTACK_FAIL)
    print_(" > Metric 1: avg. exit <<")
    print_("  - clean : {:.2f}".format(avg_exit(orig_exit_stats)))
    print_("  - attack: {:.2f}".format(avg_exit(pert_exit_stats)))
    print_(" > Metric 2: efficacy  <<")
    print_("  - clean : {:.2f}".format(efficacy(orig_exit_stats)))
    print_("  - attack: {:.2f}".format(efficacy(pert_exit_stats)))
    print_("\n")

    print_(">> Attack results (success) <<")
    orig_exit_stats, pert_exit_stats = layer_info(results, ATTACK_SUCC)
    print_(" > Metric 1: avg. exit <<")
    print_("  - clean : {:.2f}".format(avg_exit(orig_exit_stats)))
    print_("  - attack: {:.2f}".format(avg_exit(pert_exit_stats)))
    print_(" > Metric 2: efficacy  <<")
    print_("  - clean : {:.2f}".format(efficacy(orig_exit_stats)))
    print_("  - attack: {:.2f}".format(efficacy(pert_exit_stats)))
    print_("\n")

    print_(">> Attack results (skipped) <<")
    orig_exit_stats, pert_exit_stats = layer_info(results, ATTACK_SKIP)     # skipped results
    print_(" > Metric 1: avg. exit <<")
    print_("  - clean : {:.2f}".format(avg_exit(orig_exit_stats)))
    print_("  - attack: {:.2f}".format(avg_exit(pert_exit_stats)))
    print_(" > Metric 2: efficacy  <<")
    print_("  - clean : {:.2f}".format(efficacy(orig_exit_stats)))
    print_("  - attack: {:.2f}".format(efficacy(pert_exit_stats)))
    print_("\n")

    print_(">> Attack results (total) <<")
    orig_exit_stats, pert_exit_stats = layer_info(results, ATTACK_ALLS)     # success, fail, and skipped
    print_(" > Metric 1: avg. exit <<")
    print_("  - clean : {:.2f}".format(avg_exit(orig_exit_stats)))
    print_("  - attack: {:.2f}".format(avg_exit(pert_exit_stats)))
    print_(" > Metric 2: efficacy  <<")
    print_("  - clean : {:.2f}".format(efficacy(orig_exit_stats)))
    print_("  - attack: {:.2f}".format(efficacy(pert_exit_stats)))
    print_("\n")
    
    # fin.


def layer_info(results, attack_status, prefix=''):
    # data-holders to store results
    orig_exit_results = [0 for x in range(MAX_LAYERS+1)]
    pert_exit_results = [0 for x in range(MAX_LAYERS+1)]

    orig_avg_score_from_layer = [0 for x in range(MAX_LAYERS+1)]
    pert_avg_score_from_layer = [0 for x in range(MAX_LAYERS+1)]

    # counter (to compare the stats)
    pert_stats = {
        'fail': 0,
        'succ': 0,
        'skip': 0,
        'tot.': 0,
    }

    # loop over the results
    for each_result in results:

        # : only keep the failed results
        if (attack_status == ATTACK_FAIL):
            if not each_result.perturbed_result.goal_status == GoalFunctionResultStatus.SEARCHING:
                continue
            pert_stats['fail'] += 1
        
        # : only keep the successful results
        elif (attack_status == ATTACK_SUCC):
            is_successful = (
                each_result.perturbed_result.goal_status == GoalFunctionResultStatus.SUCCEEDED or \
                each_result.perturbed_result.goal_status == GoalFunctionResultStatus.MAXIMIZING
            )
            if not is_successful: continue
            pert_stats['succ'] += 1

        # : only keep the successful or failed results
        elif (attack_status == ATTACK_SKIP):
            if (each_result.perturbed_result.goal_status != GoalFunctionResultStatus.SKIPPED):
                continue
            pert_stats['skip'] += 1

        # : just to keep track of the total
        pert_stats['tot.'] += 1

        # : exit points
        exit_orig = each_result.original_result.data.layer
        exit_pert = each_result.perturbed_result.data.layer
        orig_exit_results[exit_orig] += 1
        pert_exit_results[exit_pert] += 1

        # : scoress
        orig_score = 1 - each_result.original_result.score
        pert_score = 1 - each_result.perturbed_result.score

        orig_avg_score_from_layer[exit_orig] += orig_score
        pert_avg_score_from_layer[exit_pert] += pert_score


    # check the stats.
    print_(" > Fail / Succ / Skip / Tot.: {} / {} / {} / {}".format( \
        pert_stats['fail'], pert_stats['succ'], pert_stats['skip'], pert_stats['tot.']))


    # loop over the results
    for lidx in range(len(orig_exit_results)):

        # : compute average
        if pert_exit_results[lidx] == 0:
            pert_avg_score_from_layer[lidx] = 0
        else:
            pert_avg_score_from_layer[lidx] = pert_avg_score_from_layer[lidx] / pert_exit_results[lidx]

        if orig_exit_results[lidx] == 0:
            orig_avg_score_from_layer[lidx] = 0
        else:
            orig_avg_score_from_layer[lidx] = orig_avg_score_from_layer[lidx] / orig_exit_results[lidx]

    
    # print only there is data
    if len(results):
        print_(' > Stats (exit status)')
        print_('  - Clean : {}'.format(orig_exit_results))
        print_('  - Attack: {}'.format(pert_exit_results))
        print_(' > Stats (exit scores)')
        print_('  - Clean : {}'.format(orig_avg_score_from_layer))
        print_('  - Attack: {}'.format(pert_avg_score_from_layer))
        
    else:
        print_(" > No status")

    return orig_exit_results, pert_exit_results

# ----------------------------------------------------------------
#   Metrics.
# ----------------------------------------------------------------
def avg_exit(stats):
    # ----------------------------------------
    #   Compute avg. exit numbers
    #   (as BERTs have the same size layers)
    # ----------------------------------------
    tot_exits = 0
    for lidx in range(len(stats)):
        tot_exits += (lidx + 1) * stats[lidx]
    return tot_exits / (sum(stats) + 1e-9)

def efficacy(stats):
    # ----------------------------------------
    #   Compute the efficacy; AUC under the cumulative function
    # ----------------------------------------
    if sum(stats) != 0:
        tot_probs = np.cumsum(stats) / sum(stats)
        tot_fracs = (np.arange(len(stats)) + 1) / len(stats)
        efficacy  = np.trapz(tot_probs, tot_fracs)
    else:
        efficacy  = 0.
    return efficacy


"""
    Main (using the configurations given, perform adversarial attacks)
"""
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="Name of Glue task used for inference\
               e.g. RTE",
    )

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="bert,albert",
    )

    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="e.g. /nfs/hpc/share/ritterg/pabee/model_output/RTE_Pabee",
    )

    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=False,
        help="albert-base-v2, bert-base-uncased, some dir",
    )

    parser.add_argument(
        "--exit_type",
        default="",
        type=str,
        required=True,
        help="pabee,deebert,pastfuture",
    )

    parser.add_argument(
        "--patience",
        default="0",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--entropy",
        default=-1,
        type=float,
        required=False,
    )

    parser.add_argument(
        "--eval_reg_threshold",
        default=0.5,
        type=float,
        required=True,
    )

    parser.add_argument(
        "--regression_threshold",
        default=0.1,
        type=float,
        required=False,
    )

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="Should contain the .tsv or other files for the task \
              e.g. glue_dir used in inference",
    )

    parser.add_argument(
        "--result_dir",
        default=None,
        type=str,
        required=True,
        help="/nfs/hpc/share/ritterg/pabee/model_output/textfooler_p7_RTE_Pabee",
    )

    parser.add_argument(
        "--glue_script",
        default=None,
        type=str,
        required=True,
        help="processing script for glue dataset",
    )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    # ----------------------------------------
    #   Attack specific arguments
    # ----------------------------------------
    parser.add_argument(
        "--attack",
        default=None,
        type=str,
        required=True,
        help="fooler, bae-r, clare, gbda",
    )

    parser.add_argument(
        "--attack_goal",
        type=str,
        required=True,
        help="base / oavg / ours",
    )

    parser.add_argument(
        "--attack_method",
        type=str,
        required=True,
        help="delete / beam",
    )

    parser.add_argument(
        "--attack_maxquery",
        type=int,
        required=True,
        help="most relevant for max, layer methods, 0 is inf (still limited by a max_candidate var)",
    )

    parser.add_argument(
        "--attack_threshold",
        type=float,
        required=False,
        help="the threshold for checking the attack success (the number should be in [0, 1], only ours uses it)",
    )

    parser.add_argument(
        "--attack_num_words",
        type=int,
        required=True,
        default=3,
        help="the number of universal triggers we will use (default: 3)",
    )

    parser.add_argument(
        "--multiexit",
        action='store_true',
        help="enable this flag will result in processing a multi-exit model",
    )

    parser.add_argument(
        "--num_examples",
        default=-1,
        type=int,
        help="number of examples to attack, -1 is all",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle order of examples to attack, useful when not running on all examples",
    )
    args = parser.parse_args()
    print (json.dumps(vars(args), indent=2))

    # taskname correction
    args.task_name = args.task_name.lower()
    if(args.task_name == "sts-b"):
        args.task_name = "stsb"
    elif(args.task_name == "mnli"):
        args.task_name = "mnli-mm"

    # perform attacks
    run_attacks(args)

if __name__ == "__main__":
    main()
    # done.
