
# BERT Shouldn't Lose Its Patience

This repository contains the code for reproducing the results of our paper:

- BERT Lost Patience Won't Be Robust to Adversarial Slowdown [NeurIPS 2023]

&nbsp;

----

### TL;DR

We conduct a systematic evaluation of the robustness of multi-exit language models to adversarial slowdown and show that their computational savings aren't robust to that.

&nbsp;

### Abstract

We systematically evaluate the robustness of multi-exit language models against adversarial slowdown. To audit their robustness, we develop a slowdown attack that generates natural adversarial text bypassing early-exit points. We use the resulting Waffle attack as a vehicle to conduct a comprehensive evaluation of three multi-exit mechanisms with the GLUE benchmark against adversarial slowdown. We then show our attack significantly reduces the computational savings provided by the three methods. The more complex a mechanism is, the more vulnerable it is to adversarial slowdown. We also perform a linguistic analysis of the perturbed text inputs, identifying common perturbation patterns that our attack generates, and compare them with standard adversarial text attacks. Moreover, we show that adversarial training is ineffective in defeating our slowdown attack, but input sanitation with a conversational model, e.g., ChatGPT, can remove the perturbation effectively. This result suggests that future work is needed for developing efficient yet robust multi-exit models.

&nbsp;

----

## Prerequisites

Download the GLUE dataset from this [Google Drive link](https://drive.google.com/file/d/1E3YjvukrVRNNo06FdUTbRSf35hnks65Z/view?usp=sharing). (Or feel free to download the dataset from the Hugging Face library, below.)

```
    // unzip the file (in case you download the file)
    $ mkdir -p datasets/originals/
    $ mv GLUE.tar.gz datasets/originals
    $ cd datasets/originals/
    $ tar -zxvf GLUE.tar.gz
```

Download the dataset using the Python script `huggingface` provides.

```
    $ python download_glue_data.py
```


Download pre-trained models from this [Google Drive Link: BERTs](https://drive.google.com/file/d/1xHWlGeTq8wIjnQHsIh_UfcS6Hv-pGEam/view?usp=sharing) + [Google Drive Link: Others](https://drive.google.com/file/d/1BLd43ZECavTVb6-oyoTDJRPOooliriGb/view?usp=sharing) + [Google Drive Link: Robust](https://drive.google.com/file/d/1jfK_EWMoCt4n8IhMJJu07NtYPX2kF5fD/view?usp=share_link)

```
    // unzip the file under 'model_output' dir.
    $ mkdir -p model_output
    $ mv baseline_models.tar.gz model_output/
    $ mv baseline_models_berts.tar.gz model_output/
    $ mv robest_models.tar.gz model_output/
    $ cd model_output
    $ tar -zxvf baseline_models.tar.gz
    $ tar -zxvf baseline_models_berts.tar.gz
    $ tar -zxvf robest_models.tar.gz
```

&nbsp;

----

## Fine-tune BERT/ALBERT on the GLUE classification tasks

Note: You can skip this step if you are to use the fine-tuned models we offer.

```
    $ cd projects
    $ ./finetune.<mechanism>.sh
    // the fine-tuned models are stored under model_output dir.
```

You can run the following script for evaluating the test-time performance of fine-tuned models.

```
    $ cd projects
    $ ./eval.<mechanism>.sh
```

&nbsp;

----

## Run the adversarial attacks

We implement our code by adapting TextAttack to multi-exit models. We offer our own TextAttack repository containing the adaptation to multi-exit models. You can install this by running the following commands: 

```
    // Note: this will override TextAttack installed in your local
    $ cd TextAttack
    $ pip install -e ./
```

After you overwrite our TextAttack package, you can run the attacks by running the following commands:

```
    $ ./attack.deebert.sh
    $ ./attack.pabee.sh
    $ ./attack.pastfuture.sh
```

In the header of each script, we provide variables where you can update the attack configurations. As an example of this, to reproduce our baseline results, you can set `AGOAL` to `base` or `ours`.

&nbsp;

----

## Run the adversarial training with A2T

We adapt the existing adversarial training (A2T) to support multi-exit language models and crafting of our slowdown attacks. You can run the following script to run adversarial training:

```
    // A2T on PABEE mechanism
    $ ./finetune.pabee.at.sh
    $ ./eval.pabee.at.sh        // to test the model
```

We also support the evaluation of the defended models we construct with A2T on standard adversarial text attacks and our slowdown attacks. You can perform the evaluation by running the following script:

```
    $ ./attack.pabee.at.sh
```

&nbsp;

----

## Run manual prediction on text inputs

We perform a linguistic analysis of adversarial text inputs, and it requires running manual predictions on a list of original, perturbed, and manually sanitized samples. To run this analysis, we create the following script:

```
    $ python run_prediction.py
```

In the header of this script file, a list of configuration that you can set exists. In addition, the text samples we tested out are under `others` folder using the following name convention: `example_{dataset}.tsv`, stored in a tab separated format.

&nbsp;

----

## Run the universal slowdown attack

We also perform a universal slowdown attack by randomly selecting words and appending them to the beginning of the SST-2 dataset. All code needed to execute this attack is in a script that can be executed as follows:

```
    $ python run_universal_attack.py
```

The script is designed specifically for SST-2 (which includes hard-coded hyperparameters for training), but can be modified to work with any dataset.

&nbsp;

----

## Cite Our Work

Please cite our work if you find this source code helpful.

```
    Appear later upon acceptance.
```
