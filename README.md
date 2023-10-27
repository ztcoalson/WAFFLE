
## BERT Lost Patience Won't Be Robust to Adversarial Slowdown [NeurIPS 2023]

This repository contains the code for reproducing the results of our paper:

- [BERT Lost Patience Won't Be Robust to Adversarial Slowdown]()
- **[Zachary Coalson](mailto:coalsonz@oregonstate.edu)**, Gabriel Ritter, Rakesh Bobba, [Sanghyun Hong](https://sanghyun-hong.com).

&nbsp;

----

### TL;DR

You can use our attack to audit the vulnerability of multi-exit language models to adversarial slowdowns!

&nbsp;

### Abstract

In this paper, we systematically evaluate the robustness of multi-exit language models against adversarial slowdown. To audit their robustness, we design a slowdown attack that generates natural adversarial text bypassing early-exit points. We use the resulting WAFFLE attack as a vehicle to conduct a comprehensive evaluation of three multi-exit mechanisms with the GLUE benchmark against adversarial slowdown. We then show our attack significantly reduces the computational savings provided by the three methods in both white-box and black-box settings. The more complex a mechanism is, the more vulnerable it is to adversarial slowdown. We also perform a linguistic analysis of the perturbed text inputs, identifying common perturbation patterns that our attack generates, and comparing them with standard adversarial text attacks. Moreover, we show that adversarial training is ineffective in defeating our slowdown attack, but input sanitization with a conversational model, e.g., ChatGPT, can remove perturbations effectively. This result suggests that future work is needed for developing efficient yet robust multi-exit models.

&nbsp;

----

## Prerequisites

Download the GLUE dataset from this [Google Drive link](https://drive.google.com/file/d/1E3YjvukrVRNNo06FdUTbRSf35hnks65Z/view?usp=sharing).

```
    // unzip the file (in case you download the file)
    $ mkdir -p datasets/originals/
    $ mv GLUE.tar.gz datasets/originals
    $ cd datasets/originals/
    $ tar -zxvf GLUE.tar.gz
```

OR download the dataset using the Python script `huggingface` provides.

```
    $ python download_glue_data.py
```


Download pre-trained models from the following links: [Google Drive Link: BERTs](https://drive.google.com/file/d/1xHWlGeTq8wIjnQHsIh_UfcS6Hv-pGEam/view?usp=sharing) + [Google Drive Link: Others](https://drive.google.com/file/d/1BLd43ZECavTVb6-oyoTDJRPOooliriGb/view?usp=sharing) + [Google Drive Link: Robust](https://drive.google.com/file/d/1jfK_EWMoCt4n8IhMJJu07NtYPX2kF5fD/view?usp=share_link)

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

Install required packages.

```
    $ pip install -r requirements.txt
```

We implement our code by adapting TextAttack to multi-exit models. We offer our own TextAttack repository containing the adaptation to multi-exit models. You can install this by running the following commands: 

```
    // Note: this will override TextAttack installed in your local
    $ cd TextAttack
    $ pip install -e ./
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

## Black-box attacks

The simplest way to conduct black-box experiments is to modify the .tsv files of the datasets you wish to test with. For example, if you want to test the transferability of RTE samples crafted on PastFuture, you would format these samples into a .tsv file and rename it to the original test dataset of RTE contained in `./datasets/originals/glue_hub/RTE/test.tsv` (you would likely want to rename the original test dataset to something else to avoid overwriting it). Then, you can evaluate any model of your choosing on RTE and it will use the samples you crafted with PastFuture.

&nbsp;

----

## Cite Our Work

Please cite our work if you find this source code helpful.

```
    @inproceedings{
        Coalson2023WAFFLE,
        title={BERT Lost Patience Won't Be Robust to Adversarial Slowdown},
        author={Zachary Coalson and Gabriel Ritter and Rakesh Bobba and Sanghyun Hong},
        booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
        year={2023},
        url={https://openreview.net/forum?id=TcG8jhOPdv}
    }
```

&nbsp;

---

&nbsp;

Please contact Zachary Coalson (coalsonz@oregonstate.edu) for any questions and recommendations.
