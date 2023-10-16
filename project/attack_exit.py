"""
    A script for attacking multi-exit models
"""
# basics
from concurrent.futures import process
import os
import json
import argparse
import numpy as np

# transformers
from transformers import InputExample
from transformers import glue_processors as processors

# textattack...
import textattack
from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)
from textattack.goal_function_results.goal_function_result import (
    GoalFunctionResultStatus,
)
from textattack.constraints.pre_transformation import (
    InputColumnModification
)

# custom...
from datasets import load_dataset
from utils import _get_column_names

# custom (models)...
from pabee.modeling_pabee_albert import AlbertForSequenceClassificationWithPabee
from pabee.modeling_pabee_bert import BertForSequenceClassificationWithPabee

from deebert.modeling_highway_bert import DeeBertForSequenceClassification
from deebert.modeling_highway_roberta import DeeRobertaForSequenceClassification

from pastfuture.modeling_albert import AlbertForSequenceClassification
from pastfuture.modeling_bert import BertForSequenceClassification
from pastfuture.modeling_roberta import RobertaForSequenceClassification

# configurations
from transformers import (
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    AlbertConfig,
    AlbertTokenizer
)

# custom (other wrappers)...
from pabee_model_wrapper import *
from pastfuture_model_wrapper import *
from deebert_model_wrapper import *


# --------------------------------------------------------------------------------
#   Globals
# --------------------------------------------------------------------------------
MAX_LAYERS  = 12
ATTACK_FAIL = 0
ATTACK_SUCC = 1
ATTACK_SKIP = 2
ATTACK_ALLS = 3


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


# --------------------------------------------------------------------------------
#   Attack function
# --------------------------------------------------------------------------------
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
            "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
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
        elif args.model_type == "roberta":
            model.roberta.reset_stats()
            model.roberta.set_patience(args.entropy)
            model.roberta.multiexit=args.multiexit
            tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=False)
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
            "bert": (BertConfig, DeeBertForSequenceClassification, BertTokenizer),
            "roberta": (RobertaConfig, DeeRobertaForSequenceClassification, RobertaTokenizer),}
        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = model_classes[args.model_type]
        model = model_class.from_pretrained(args.model_path)

        if args.model_type == "bert":
            model.bert.encoder.set_early_exit_entropy(args.entropy)
            model.bert.encoder.multiexit=args.multiexit
            tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=True)
        elif args.model_type == "roberta":
            model.roberta.encoder.set_early_exit_entropy(args.entropy)
            model.roberta.encoder.multiexit=args.multiexit
            tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=False)
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


    # ----------------------------------------
    #   Modify columns depending on the data
    # ----------------------------------------
    # 1st column
    if args.task_name == "rte" or args.task_name == "wnli" or \
        args.task_name == "mnli" or args.task_name == "mnli-mm":
        input_column_modification = InputColumnModification(
            ["sentence1", "sentence2"], {"sentence2"}
        )

    # 2nd column
    elif args.task_name == "qnli":
        input_column_modification = InputColumnModification(
            ["sentence1", "sentence2"], {"sentence1"}
        )

    # one or the other (ideally though whichever one is larger)
    elif args.task_name == "stsb" or args.task_name == "qqp" or \
        args.task_name == "mrpc":
        input_column_modification = InputColumnModification(
            ["sentence1", "sentence2"], {"sentence2"}
        )

    # has one column, modify any columns, other cases (cola, sst2)
    else:
        input_column_modification = InputColumnModification(
            [], {}
        )


    # ----------------------------------------
    #   Configure an attack to use
    # ----------------------------------------
    if args.attack == "fooler":
        attack = textattack.attack_recipes.TextFoolerJin2019.build(
            model_wrapper, input_column_modification, args)
    elif args.attack == "bae-r": 
        attack = textattack.attack_recipes.BAEGarg2019.build(
            model_wrapper)
    elif args.attack == "clare":
        attack = textattack.attack_recipes.CLARE2020.build(
            model_wrapper, input_column_modification, args, args.multiexit)     # FIXME
    elif args.attack == "a2t":
        attack = textattack.attack_recipes.A2TYoo2021.build(
            model_wrapper, input_column_modification, args)
    else:
        assert False, ('Error: unsupported attack - {}'.format(args.attack))


    # attack configurations
    attack.goal_function.batch_size = 1         # Note: batch size = 1 for early-exits
    attack.goal_function.regression_threshold = args.eval_reg_threshold
    attack.goal_function.use_cache = True


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

    # depending on where we download the data
    if (args.data_dir == "glue_hub"):
        dataset = textattack.datasets.HuggingFaceDataset("glue", subset=_task, split=_split)

    # from glue.py
    else:
        processor = processors[args.task_name]()
        dataset_ = load_dataset(
            args.glue_script, 
            _task, 
            data_files=_data, 
            cache_dir=None, 
            split=_split)
        inputs, output = _get_column_names(dataset_[0], "dev", processor)
        dataset = textattack.datasets.HuggingFaceDataset(dataset_, dataset_columns=(inputs, output))
    

    # store the configurations to the attack arguments
    attack_args = textattack.AttackArgs(
        num_examples=args.num_examples,
        log_to_csv=None,
        disable_stdout=False,
        shuffle=args.shuffle,
    )

    
    # ----------------------------------------
    #   Run attacks
    # ----------------------------------------
    attacker = textattack.Attacker(attack, dataset, attack_args)
    results  = attacker.attack_dataset()


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
