from transformers import InputExample
from datasets import load_dataset


def _get_data_runglue(task, args, evaluate, processor):
    task_ = task
    if task.lower() == "sst-2":
         task_ = "sst2"
    if task.lower() == "sts-b":
         task_ = "stsb"
    if "glue_hub" == args.data_dir:
        if evaluate:
            split_ = 'validation'
            #or do task mnli and splits 'validation_matched', 'validation_mismatched'
            if task.lower() == 'mnli':
                split_ = 'validation'
                task_ = 'mnli_matched'
            if task.lower() == 'mnli-mm':
                split_ = 'validation'
                task_ = 'mnli_mismatched'
            examples = _create_examples_hub( 
                load_dataset('glue', task_.lower(), split=split_),"dev", processor)
        else:
            examples = _create_examples_hub(
                load_dataset('glue', task_.lower(), split='train'),"train", processor)
    elif args.glue_script:
        if evaluate:
            split_ = 'validation'
            #or do task mnli and splits 'validation_matched', 'validation_mismatched'
            if task.lower() == 'mnli':
                split_ = 'validation'
                task_ = 'mnli_matched'
            if task.lower() == 'mnli-mm':
                split_ = 'validation'
                task_ = 'mnli_mismatched'
            examples = _create_examples_hub( 
                # load_dataset(args.glue_script, task_, data_files=args.data_dir+"/script-dev.tsv", cache_dir=None, split=split_),"dev", processor)
                load_dataset(args.glue_script, task_, data_files=args.data_dir+"/dev.tsv", cache_dir=None, split=split_),"dev", processor)
        else:
            split_ = 'train'
            if task.lower() == 'mnli-mm':
                task_='mnli'
            examples = _create_examples_hub(
                # load_dataset(args.glue_script, task_, data_files=args.data_dir+"/script-dev.tsv", cache_dir=None, split=split_),"train", processor)
                load_dataset(args.glue_script, task_, data_files=args.data_dir+"/dev.tsv", cache_dir=None, split=split_),"train", processor)
    else:
        #plaintext like the tsv file
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )

    return examples

def _create_examples_hub(lines,set_type, processor):
        """Creates examples for the training, dev and test sets from huggingface hub, not tsv"""
        examples = []
        keys = lines[0].keys()
        idx=True
        s_=True
        s1=True
        s2=True
        lab=True
        score=True
        q_ = True
        q1 = True
        q2 = True
        p = True
        h = True

        if "idx" not in keys: idx = False
        if "sentence" not in keys: s_ = False
        if "sentence1" not in keys: s1 = False
        if "sentence2" not in keys: s2 = False
        if "question" not in keys: q_ = False
        if "question1" not in keys: q1 = False
        if "question2" not in keys: q2 = False
        if "premise" not in keys: p = False
        if "hypothesis" not in keys: h = False

        labels = processor.get_labels()
        if "label" not in keys: lab = False
        if "score" not in keys: score = False
        for line in lines:
            guid=line["idx"] if idx else None
            text_a=None
            text_b=None
            if q1 and q2:
                text_a=line["question1"]
                text_b=line["question2"]
            elif s_ and not q_:
                text_a=line["sentence"]
                text_b=None
            elif s_ and q_:
                text_a=line["question"]
                text_b=line["sentence"]
            elif s1 and s2:
                text_a=line["sentence1"]
                text_b=line["sentence2"]
            elif p and h:
                text_a=line["premise"]
                text_b=line["hypothesis"]
            if lab:
                if labels[0]:
                    label = labels[line["label"]]
                else:
                    label = line["label"]
            elif score:
                if labels[0]:
                    label = labels[line["score"]]
                else:
                    label = line["score"]
            if set_type == "test": label = None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def _get_column_names(line, set_type, processor):
        """Creates examples for the training, dev and test sets from huggingface hub, not tsv"""
        keys = line.keys()
        idx=True
        s_=True
        s1=True
        s2=True
        lab=True
        score=True
        q_ = True
        q1 = True
        q2 = True
        p = True
        h = True

        if "idx" not in keys: idx = False
        if "sentence" not in keys: s_ = False
        if "sentence1" not in keys: s1 = False
        if "sentence2" not in keys: s2 = False
        if "question" not in keys: q_ = False
        if "question1" not in keys: q1 = False
        if "question2" not in keys: q2 = False
        if "premise" not in keys: p = False
        if "hypothesis" not in keys: h = False

        labels = processor.get_labels()
        if "label" not in keys: lab = False
        if "score" not in keys: score = False
            
        guid= "idx" if idx else None
        text_a=None
        text_b=None
        if q1 and q2:
            text_a="question1"
            text_b="question2"
        elif s_ and not q_:
            text_a="sentence"
            text_b=None
        elif s_ and q_:
            text_a="question"
            text_b="sentence"
        elif s1 and s2:
            text_a="sentence1"
            text_b="sentence2"
        elif p and h:
            text_a="premise"
            text_b="hypothesis"
        if lab:
            label = "label"
        elif score:
            label = "score"
        if set_type == "test": label = None

        # post-process, to remove the None
        inputs = [text_a, text_b]
        inputs = [each_text for each_text in inputs if each_text]
        output = label
        return inputs, output

def main():
    pass

if __name__ == "__main__":
    main()