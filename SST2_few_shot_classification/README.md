# LM-BFF (**B**etter **F**ew-shot **F**ine-tuning of **L**anguage **M**odels)

This is the implementation of the paper [Making Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/pdf/2012.15723.pdf). LM-BFF is short for **b**etter **f**ew-shot **f**ine-tuning of **l**anguage **m**odels.

## Table of contents

- [LM-BFF (**B**etter **F**ew-shot **F**ine-tuning of **L**anguage **M**odels)](#lm-bff-better-few-shot-fine-tuning-of-language-models)
  - [Table of contents](#table-of-contents)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Prepare the data](#prepare-the-data)
  - [Run LM-BFF](#run-lm-bff)
    - [Quick start](#quick-start)
    - [How to design your own templates](#how-to-design-your-own-templates)
  - [Citation](#citation)


## Overview

In this work we present LM-BFF, a suite of simple and complementary techniques for fine-tuning pre-trained language models on a small number of training examples. Our approach includes:

    Prompt-based fine-tuning together with a novel pipeline for automating prompt generation.

You can find more details of this work in our [paper](https://arxiv.org/pdf/2012.15723.pdf).

## Requirements

To run our code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```
## Prepare the data

For my implementation I already included <b>SST-2</b> dataset in <b>data/original/SST-2/</b> directory.

Then use the following command (in the root directory) to generate the few-shot data we need:

```bash
python tools/generate_k_shot_data.py
```

See `tools/generate_k_shot_data.py` for more options. For results in the paper, we use the default options: we take `K=16` and take 5 different seeds of 13, 21, 42, 87, 100. The few-shot data will be generated to `data/k-shot`. In the directory of each dataset, there will be folders named as `$K-$SEED` indicating different dataset samples. You can use the following command to check whether the generated data are exactly the same as ours:

```bash
cd data/k-shot
md5sum -c checksum
```

**NOTE**: During training, the model will generate/load cache files in the data folder. If your data have changed, make sure to clean all the cache files (starting with "cache").

## Run LM-BFF

### Quick start
Our code is built on [transformers](https://github.com/huggingface/transformers). We did all our experiments with version `3.4.0`, but it should work with other versions.

Before running any experiments, create the result folder by `mkdir result` to save checkpoints. Then you can run our code with the following example:

```bash
roberta-base model and seed=42:-
------------------------------------
python run.py --task_name='SST-2' --cache_dir="model_cache_dir_rob_base" --data_dir='data/k-shot/SST-2/16-42' --overwrite_output_dir --do_train --do_eval --do_predict --model_name_or_path="roberta-base" --few_shot_type="prompt-demo" --num_k=16 --max_steps=1000 --eval_steps=100 --per_device_train_batch_size=2 --learning_rate=1e-5 --num_train_epochs=0 --output_dir='result/tmp' --seed=42 --template="*cls**sent_0*_It_was*mask*.*sep+*" --mapping="{'0':'terrible','1':'great'}" --num_sample=16 --evaluation_strategy="steps"

bert-base-uncased model and seed=42:-
------------------------------------
python run.py --task_name='SST-2' --cache_dir="model_cache_dir_bert_base" --data_dir='data/k-shot/SST-2/16-42' --overwrite_output_dir --do_train --do_eval --do_predict --model_name_or_path="bert-base-uncased" --few_shot_type="prompt-demo" --num_k=16 --max_steps=1000 --eval_steps=100 --per_device_train_batch_size=2 --learning_rate=1e-5 --num_train_epochs=0 --output_dir='result/tmp' --seed=42 --template="*cls**sent_0*_It_was*mask*.*sep+*" --mapping="{'0':'terrible','1':'great'}" --num_sample=16 --evaluation_strategy="steps"
```

Most arguments are inherited from `transformers` and are easy to understand. We further explain some of the LM-BFF's arguments:

* `few_shot_type`: There are three modes
  * `finetune`: Standard fine-tuning
  * `prompt`: Prompt-based fine-tuning.
  * `prompt-demo`: Prompt-based fine-tuning with demonstrations.
* `num_k`: Number of training instances for each class. We take `num_k`=16 in our paper. This argument is mainly used for indexing logs afterwards (because the training example numbers are actually decided by the data split you use).
* `template`: Template for prompt-based fine-tuning. We will introduce the template format later.
* `mapping`: Label word mapping for prompt-based fine-tuning. It is a string of dictionary indicating the mapping from label names to label words. **NOTE**: For RoBERTa, the model will automatically add space before the word. See the paper appendix for details.
* `num_sample`: When using demonstrations during inference, the number of samples for each input query. Say `num_sample`=16, then we sample 16 different sets of demonstrations for one input, do the forward seperately, and average the logits for all 16 samples as the final prediction.

Also, this codebase supports BERT-series and RoBERTa-series pre-trained models in Huggingface's `transformers`. You can check [Huggingface's website](https://huggingface.co/models) for available models and pass models with a "bert" or "roberta" in their names to `--model_name_or_path`. Some examples would be `bert-base-uncased`, `bert-large-uncased`, `roberta-base`, `roberta-large`, etc.

To easily run our experiments, you can also use `run_experiment.sh` (this command runs prompt-based fine-tuning with demonstrations, no filtering, manual prompt):

```bash
TAG=exp TYPE=prompt-demo TASK=SST-2 BS=2 LR=1e-5 SEED=42 MODEL=roberta-large bash run_experiment.sh
```

We have already defined the templates and label word mappings in it, so you only need manipulate several hyper-parameters and `TAG` (you can use whatever tag you want and it just makes finding results easier). See `run_experiment.sh` for more options of these environment variables. Besides, you can add extra arguments by

```bash
TAG=exp TYPE=prompt-demo TASK=SST-2 BS=2 LR=1e-5 SEED=42 MODEL=roberta-large bash run_experiment.sh "--output_dir result/exp --max_seq_length 512"
```


### How to design your own templates

Here are two template examples:

For SST-2: `*cls**sent_0*_It_was*mask*.*sep+*` => `[CLS] {S0} It was [MASK]. [SEP]`

For MNLI: `*cls**sent-_0*?*mask*,*+sentl_1**sep+*` => `[CLS] {S0}? [MASK], {S1} [SEP]`

The template is composed of special tokens and variables (surrounded by `*`) and text (e.g., `It_was`, where space is replaced by `_`). Special tokens and variables contain:

* `*cls*`, `*sep*`, `*sep+*` and `*mask*`: Special tokens of CLS, SEP and MASK (different for different pre-trained models and tokenizers). `*sep+*` means the contents before and after this token have different segment embeddings (only for BERT).
* `*sent_i*`: The i-th sentence.
* `*sent-_i*`: The i-th sentence, discarding the last character.
* `*sentl_i*`: The i-th sentence, lower-casing the first letter.
* `*sentl-_i*`: The i-th sentence, discarding the last character and lower-casing the first letter.
* `*+sent_i*`: The i-th sentence, adding an extra space at the beginning.
* `*+sentl_i*`: The i-th sentence, adding an extra space at the beginning and lower-casing the first letter.

## Citation

Please cite our paper if you use LM-BFF in your work:

```bibtex
@inproceedings{gao2021making,
   title={Making Pre-trained Language Models Better Few-shot Learners},
   author={Gao, Tianyu and Fisch, Adam and Chen, Danqi},
   booktitle={Association for Computational Linguistics (ACL)},
   year={2021}
}
```
