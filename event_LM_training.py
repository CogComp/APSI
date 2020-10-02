"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
import ujson as json
from util import *
# from transformers import (
#     CONFIG_MAPPING,
#     MODEL_WITH_LM_HEAD_MAPPING,
#     AutoConfig,
#     AutoModelWithLMHead,
#     AutoTokenizer,
#     DataCollatorForLanguageModeling,
#     HfArgumentParser,
#     LineByLineTextDataset,
#     PreTrainedTokenizer,
#     TextDataset,
#     Trainer,
#     TrainingArguments,
#     set_seed,
# )

from transformers import *
import torch
from torch.nn.utils.rnn import pad_sequence
from random import choice


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default='roberta-base',
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default='roberta-base', metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default='roberta-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='/', metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=True, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False, local_rank=-1):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)


def evaluate_probability_of_event_sequence(model, tokenizer, device, event_sequence):
    model.to(device)

    tmp_sequence = '[CLS] '
    for tmp_subevent in event_sequence:
        all_words = tmp_subevent.split('$$')[1:]
        subject = list()
        other_words = list()
        for tmp_w in all_words:
            if tmp_w.split(':')[0] == 'ROOT$nsubj':
                subject.append(tmp_w.split(':')[1])
            else:
                other_words.append(tmp_w.split(':')[1])
        all_clean_words = subject + other_words
        tmp_event = ' '.join(all_clean_words)
        tmp_sequence += tmp_event
    tmp_sequence += ' [SEP]'

    test_example = pad_sequence([torch.tensor(tokenizer.encode(tmp_sequence))], batch_first=True).to(device)

    # print(test_example)
    outputs = model(input_ids=test_example, masked_lm_labels=test_example)
    # print(outputs)

    return outputs[0].mean().item()

def event_sequence_to_training_example(input_sequence, tokenizer):
    tmp_sequence = '[CLS] '
    for tmp_subevent in input_sequence:
        all_words = tmp_subevent.split('$$')[1:]
        subject = list()
        other_words = list()
        for tmp_w in all_words:
            if tmp_w.split(':')[0] == 'ROOT$nsubj':
                subject.append(tmp_w.split(':')[1])
            else:
                other_words.append(tmp_w.split(':')[1])
        all_clean_words = subject + other_words
        tmp_event = ' '.join(all_clean_words)
        tmp_sequence += tmp_event
    tmp_sequence += ' [SEP]'
    return torch.tensor(tokenizer.encode(tmp_sequence))

def tokenize_event_sequence_with_mask(tokenizer, event_sequence, target_position):
    tokenized_sequence = list()
    labels = list()
    for i, tmp_subevent in enumerate(event_sequence):
        all_words = tmp_subevent.split('$$')[1:]
        subject = list()
        other_words = list()
        for tmp_w in all_words:
            if tmp_w.split(':')[0] == 'ROOT$nsubj':
                subject.append(tmp_w.split(':')[1])
            else:
                other_words.append(tmp_w.split(':')[1])
        all_clean_words = subject + other_words
        tmp_event = ' '.join(all_clean_words)
        tmp_event = '[CLS] ' + tmp_event + ' [SEP] '
        tmp_encoded_event = tokenizer.encode(tmp_event)
        for tmp_id in tmp_encoded_event:
            if i == target_position:
                labels.append(tmp_id)
                tokenized_sequence += tokenizer.convert_tokens_to_ids(['[MASK]'])
            else:
                labels.append(-100)
                tokenized_sequence.append(tmp_id)
    return tokenized_sequence, labels


def divide_chunks(l, n):
    new_list = list()
    for i in range(0, len(l), n):
        new_list.append(l[i:i + n])
    return new_list


def tokenize_event_sequence_with_random_position(tokenizer, event_sequences):
    tokenized_event_sequences = list()
    tokenized_labels = list()
    for tmp_event_sequence in event_sequences:
        tmp_tokenized_event_sequence, tmp_tokenized_label = tokenize_event_sequence_with_mask(tokenizer,
                                                                                              tmp_event_sequence,
                                                                                              choice(range(len(
                                                                                                  tmp_event_sequence))))
        tokenized_event_sequences.append(torch.tensor(tmp_tokenized_event_sequence))
        tokenized_labels.append(torch.tensor(tmp_tokenized_label))
    return tokenized_event_sequences, tokenized_labels


def tokenize_event_sequence_with_position(tokenizer, event_sequences, position):
    tokenized_event_sequences = list()
    tokenized_labels = list()
    for tmp_event_sequence in event_sequences:
        tmp_tokenized_event_sequence, tmp_tokenized_label = tokenize_event_sequence_with_mask(tokenizer,
                                                                                              tmp_event_sequence,
                                                                                              position)
        tokenized_event_sequences.append(torch.tensor(tmp_tokenized_event_sequence))
        tokenized_labels.append(torch.tensor(tmp_tokenized_label))
    return tokenized_event_sequences, tokenized_labels


def train_masked_event_LM(model, device, training_args, tokenized_event_sequences, tokenized_labels):

    batch_size = 8

    chunked_input = divide_chunks(tokenized_event_sequences, batch_size)
    chunked_label = divide_chunks(tokenized_labels, batch_size)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=len(chunked_input)
    )
    total_loss = 0
    for i in tqdm(range(len(chunked_input)), desc='step'):
        tensorized_event_sequence = pad_sequence(chunked_input[i], batch_first=True).to(device)
        tensorized_labels = pad_sequence(chunked_label[i], batch_first=True).to(device)

        model.train()

        outputs = model(input_ids=tensorized_event_sequence, masked_lm_labels=tensorized_labels)
        loss = outputs[0]
        loss.backward()
        tmp_loss = loss.item()
        total_loss += tmp_loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.num_train_epochs = 50
    training_args.save_steps = 100000
    training_args.logging_steps = 500
    set_seed(training_args.seed)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    training_sequences = list()
    with open('intrinsic_dataset/train_processes.json', 'r') as f:
        test_processes = json.load(f)
        for tmp_process in test_processes:
            training_sequences.append(tmp_process['subevent_structures'])

    model.to(training_args.device)
    for i in tqdm(range(50), desc='iteration'):
        random.shuffle(training_sequences)
        tokenized_event_sequences, tokenized_labels = tokenize_event_sequence_with_random_position(tokenizer,
                                                                                                   training_sequences)
        train_masked_event_LM(model, training_args.device, training_args, tokenized_event_sequences, tokenized_labels)

    model.save_pretrained(training_args.output_dir)
    torch.save(training_args, os.path.join(training_args.output_dir, "training_args.bin"))
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()