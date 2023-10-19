"""
    A script for analyzing the custom texts
    (Please use the configurations section for your use)
"""
# basics
from concurrent.futures import process
import os
import csv
import numpy as np

# disable all the warnings, too much
import warnings
warnings.filterwarnings("ignore")

# torch
from torch.utils.data import DataLoader, TensorDataset

# transformers
from transformers import InputExample
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from transformers import glue_convert_examples_to_features as convert_examples_to_features

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


# --------------------------------------------------------------------------------
#   Configurations
# --------------------------------------------------------------------------------
# input-specific
_task_name = 'CoLA'                                  # RTE, MNLI, MRPC, QNLI, QQP, SST-2, CoLA
_data_file = os.path.join(
    '..', 'others', 
    'example_{}.tsv'.format(_task_name.lower()))     # edit this file and add custom inputs
"""
    Note: the above file is a 'tab' separated file; 
          check if each column is correctly separated with a tab
"""

# mechanism-specific
_multiexit = True
_mechanism = 'pabee'                                # deebert, pabee, pastfuture
_threshold = 6                                      # patience 6, entropy: refer the file

# model-specific
_modelbase = 'albert'                               # bert, albert
_modelname = 'albert-base-v2'                       # bert-base-uncased, albert-base-v2
_modelpath = '../model_output/{}_{}_{}'.format( \
    _modelbase, _mechanism if _mechanism != 'pabee' else _mechanism.upper(), _task_name)
_maxlength = 128

# misc.
_reg_thres = 0.1





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
#   Implement our own data separator
# --------------------------------------------------------------------------------
def _load_tsv_data(input_file, quotechar=None):
    # load the tab separated file
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

def _create_examples(task, lines):
    # data-holder
    examples = []

    # for each task
    if task == 'rte':
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid   = f"test-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            label  = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    elif task == 'mnli-mm':
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid   = f"test-{line[0]}"
            text_a = line[8]
            text_b = line[9]
            label  = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    elif task == 'mrpc':
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid   = f"test-{i}"
            text_a = line[3]
            text_b = line[4]
            label  = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    elif task == 'qnli':
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid   = f"test-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            label  = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    elif task == 'qqp':
        q1_index = 1
        q2_index = 2
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f"test-{line[0]}"
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label  = line[3]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    elif task == 'sst-2':
        text_index = 0
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid   = f"test-{i}"
            text_a = line[text_index]
            label  = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            print (examples[-1])
    elif task == 'cola':
        text_index = 2
        examples = []
        for i, line in enumerate(lines):
            guid   = f"test-{i}"
            text_a = line[text_index]
            label  = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    else:
        assert False, ('Error: unsupported task name - {}, abort.'.format(task))

    # return the data
    return examples
    
def _load_data_from_file(task, datafile):
    # load data from a file
    examples = _create_examples(task, _load_tsv_data(datafile))
    return examples

def load_and_cache_examples(task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]

    # load the labels 
    label_list = processor.get_labels()

    # load the custom data
    examples = _load_data_from_file(task, _data_file)

    # converted, like used in the dataloader batch iteration
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=_maxlength,
        output_mode=output_mode,
    )

    # convert them to Tensors and build a dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    all_indexes    = torch.tensor([i for i in range(len(features))], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, 
        all_attention_mask, 
        all_token_type_ids, 
        all_labels, 
        all_indexes)
    return dataset, examples



# --------------------------------------------------------------------------------
#   Prediction function
# --------------------------------------------------------------------------------
def run_prediction():

    # ----------------------------------------
    #   load a multi-exit model
    # ----------------------------------------
    # print out the configurations
    print (' : load a model with an early-exit mechanism')
    print ('  - base     : {}'.format(_modelbase))
    print ('  - base-name: {}'.format(_modelname))
    print ('  - mechanism: {}'.format(_mechanism))
    print ('  - threshold: {}'.format(_threshold))
    print ('  - modelfile: {}'.format(_modelpath))

    # load models (PABEE)
    if _mechanism == 'pabee':
        model_classes = {
            "bert": (BertConfig, BertForSequenceClassificationWithPabee, BertTokenizer),
            "albert": (AlbertConfig, AlbertForSequenceClassificationWithPabee, AlbertTokenizer),}
        _modeltype = _modelbase.lower()
        config_class, model_class, tokenizer_class = model_classes[_modeltype]
        model = model_class.from_pretrained(_modelpath)

        if _modeltype == "albert":
            model.albert.set_regression_threshold(_reg_thres)
            model.albert.set_patience(int(_threshold))
            model.albert.reset_stats()
            tokenizer = tokenizer_class.from_pretrained(_modelpath, do_lower_case=True)
            model.albert.multiexit=_multiexit
        elif _modeltype == "bert":
            model.bert.set_regression_threshold(_reg_thres)
            model.bert.set_patience(int(_threshold))
            model.bert.reset_stats()
            tokenizer = tokenizer_class.from_pretrained(_modelpath, do_lower_case=True)
            model.bert.multiexit=_multiexit
        else:
            raise NotImplementedError()
        
        # : use the wrapper
        model_wrapper = PabeeModelWrapper( \
            model, tokenizer, \
            _maxlength, \
            _multiexit, \
            'none')         # this is attack goal, but it doesn't matter in this script

    # load models (PASTFUTURE)
    elif _mechanism == 'pastfuture':
        model_classes = {
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
            "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),}
        _modeltype = _modelbase.lower()
        config_class, model_class, tokenizer_class = model_classes[_modeltype]
        model = model_class.from_pretrained(_modelpath)

        if _modeltype == "albert":
            model.albert.reset_stats()
            model.albert.set_patience(_threshold)
            model.albert.multiexit=_multiexit
            tokenizer = tokenizer_class.from_pretrained(_modelpath, do_lower_case=True)
        elif _modeltype == "bert":
            model.bert.reset_stats()
            model.bert.set_patience(_threshold)
            model.bert.multiexit=_multiexit
            tokenizer = tokenizer_class.from_pretrained(_modelpath, do_lower_case=True)
        else:
            raise NotImplementedError()

        # : use the wrapper
        model_wrapper = PastfutureModelWrapper( \
            model, tokenizer, \
            _maxlength, \
            _modeltype, \
            _multiexit, \
            'none')         # this is attack goal, but it doesn't matter in this script
        
    # load models (DEEBERT)
    elif _mechanism == 'deebert':
        model_classes = {
            "bert": (BertConfig, DeeBertForSequenceClassification, BertTokenizer),}
        _modeltype = _modelbase.lower()
        config_class, model_class, tokenizer_class = model_classes[_modeltype]
        model = model_class.from_pretrained(_modelpath)

        if _modeltype == "bert":
            model.bert.encoder.set_early_exit_entropy(_threshold)
            model.bert.encoder.multiexit=_multiexit
            tokenizer = tokenizer_class.from_pretrained(_modelpath, do_lower_case=True)
        else:
            raise NotImplementedError()
    
        # : use the wrapper
        model_wrapper = DeebertModelWrapper( \
            model, tokenizer, \
            _maxlength, \
            _multiexit, \
            'none')         # this is attack goal, but it doesn't matter in this script

    # others (raise an error...)
    else:
        raise NotImplementedError()


    # ----------------------------------------
    #   compose a dataset with our custom data
    # ----------------------------------------
    custom_dataset, custom_examples = \
        load_and_cache_examples(_task_name, tokenizer)
    custom_dataloder = DataLoader(
        custom_dataset, 
        shuffle=False,      # do not shuffle
        batch_size=1)       # multi-exit models use the batch size of 1
    print (' : load the custom data')
    print ('   - file: {}'.format(_data_file))
    print ('   - len.: {}'.format(len(custom_dataset)))


    # ----------------------------------------
    #   Report the results over the data
    # ----------------------------------------
    for bidx, (each_batch, each_example) in \
        enumerate(zip(custom_dataloder, custom_examples)):
        # : set the model to eval-mode
        model.eval()

        # : decompose the batch
        inputs = {
            "input_ids": each_batch[0],
            "attention_mask": each_batch[1],
            "labels": each_batch[3],
        }
        inputs["token_type_ids"] = each_batch[2]
        
        print ('----------------------------------------')
        print (': [{}-th] custom sample'.format(bidx))
        print ('  - text 1: {:40s}'.format(each_example.text_a))
        if ('sst' not in _task_name) \
            and ('cola' not in _task_name):
            print ('  - text 2: {:40s}'.format(each_example.text_b))
        print ('  - label : {}'.format(each_example.label))        
    
        # : run forward
        if _mechanism == 'pabee':
            outputs, num_flayer, num_elayer, all_logits = model(**inputs)
            tot_loss, exit_logit = outputs
            print ('  - exit #: {} (pabee)'.format(num_elayer.item()))
        
        elif _mechanism == 'pastfuture':
            outputs, all_logits = model(**inputs)
            num_elayer = eval('model.{}.exit_layer'.format(_modelbase))
            print ('  - exit #: {} (pastfuture)'.format(num_elayer))

        elif _mechanism == 'deebert':
            outputs = model(**inputs)
            num_elayer = outputs[-1]
            print ('  - exit #: {} (deebert)'.format(num_elayer))

        else:
            assert False, ('Error: unsupported multi-exit mechanism - {}, abort'.format(_mechanism))


    print (' : done.')
    # fin.


"""
    Main (to run the prediction with custom input data)
"""
if __name__ == "__main__":

    # taskname correction, to use the transformers preprocessor
    _task_name = _task_name.lower()
    if _task_name == 'sts-b':
        _task_name = "stsb"
    elif _task_name == "mnli":
        _task_name = "mnli-mm"

    # run the prediction with a single input
    # modify the configurations in the header
    run_prediction()
    # done.
