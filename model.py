"""
In this file we store all class for the models 
tensorflow 2.0

{
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "do_sample": false,
  "eos_token_ids": 0,
  "finetuning_task": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1"
  },
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "is_decoder": false,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1
  },
  "layer_norm_eps": 1e-12,
  "length_penalty": 1.0,
  "max_length": 20,
  "max_position_embeddings": 512,
  "num_attention_heads": 16,
  "num_beams": 1,
  "num_hidden_layers": 24,
  "num_labels": 2,
  "num_return_sequences": 1,
  "output_attentions": false,
  "output_hidden_states": false,
  "output_past": true,
  "pad_token_id": 0,
  "pruned_heads": {},
  "repetition_penalty": 1.0,
  "temperature": 1.0,
  "top_k": 50,
  "top_p": 1.0,
  "torchscript": false,
  "type_vocab_size": 2,
  "use_bfloat16": false,
  "vocab_size": 30522
}
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

import argparse
import os
import random
import time
import pickle
import math
from collections import namedtuple
import tensorflow as tf
from transformers import TFBertMainLayer, TFBertPreTrainedModel, TFRobertaMainLayer, TFRobertaPreTrainedModel, \
    TFAlbertPreTrainedModel
from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer, AlbertTokenizer, AlbertConfig
from modeling_tf_albert import TFALBertMainLayer
from transformers.modeling_tf_utils import get_initializer
import dataset_utils
from time import time


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time() - self.starttime)


class TFBertForNaturalQuestionAnswering(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = TFBertMainLayer(config, name='bert')
        self.initializer = get_initializer(config.initializer_range)

        # after we have the bert embeddings we calculate the start token with a fully connected 
        self.layer_1 = tf.keras.layers.Dense(512,
                                             kernel_initializer=self.initializer, activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(256,
                                            kernel_initializer=self.initializer, activation=tf.nn.relu)
        self.start = tf.keras.layers.Dense(1,
                                           kernel_initializer=self.initializer, name="start", activation=tf.nn.softmax)

        self.end = tf.keras.layers.Dense(1,
                                         kernel_initializer=self.initializer, name="end", activation=tf.nn.softmax)

        self.type = tf.keras.layers.Dense(5, kernel_initializer=self.initializer,
                                          activation=tf.nn.softmax, name="type")

    def call(self, inputs, **kwargs):
        bert_output = self.bert(inputs)
        presoftmax = self.layer2(self.layer_1(bert_output[0]))

        start = self.start(presoftmax)
        end = self.end(presoftmax)
        answer_type = self.type(bert_output[1])

        return start, end, answer_type


class TFAlbertForNaturalQuestionAnswering(TFAlbertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.albert = TFALBertMainLayer(config, name='albert')
        self.initializer = get_initializer(config.initializer_range)

        # after we have the bert embeddings we calculate the start token with a fully connected
        self.layer_1 = tf.keras.layers.Dense(512,
                                             kernel_initializer=self.initializer, activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(256,
                                            kernel_initializer=self.initializer, activation=tf.nn.relu)
        self.start = tf.keras.layers.Dense(1,
                                           kernel_initializer=self.initializer, name="start", activation=tf.nn.softmax)

        self.end = tf.keras.layers.Dense(1,
                                         kernel_initializer=self.initializer, name="end", activation=tf.nn.softmax)

        self.type = tf.keras.layers.Dense(5, kernel_initializer=self.initializer,
                                          activation=tf.nn.softmax, name="type")

    def call(self, inputs, **kwargs):
        bert_output = self.albert(inputs)
        presoftmax = self.layer2(self.layer_1(bert_output[0]))

        start = self.start(presoftmax)
        end = self.end(presoftmax)
        answer_type = self.type(bert_output[1])

        return start, end, answer_type


# model.compile(optimizer, loss = losses, loss_weights = lossWeights)


def main(namemodel, batch_size, namefile,  verbose=False, evaluate=False, max_num_samples=1_000_000):
    """

    :param namemodel: nomde del modello da eseguire
    :param batch_size: dimensione del batch durante il training
    :param namefile: path del file da allenare/valutare
    :param verbose: fag per stampare informazioni sul primo elemento del dataset
    :param evaluate: Bool per indicare se dobbiamo eseguire Evaluation o Training. Training di Default
    :param max_num_samples: massimo numero di oggetti da prendere in considerazione (1mil Default)
    :return: TUTTO
    """
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                     histogram_freq=1,
                                                     profile_batch='500,520')
    MODEL_CLASSES = {
        'bert': (BertConfig, TFBertForNaturalQuestionAnswering, BertTokenizer),
        'albert': (AlbertConfig, TFAlbertForNaturalQuestionAnswering, AlbertTokenizer),  # V2
        # 'roberta': (RobertaConfig, TFRobertaForNaturalQuestionAnswering, RobertaTokenizer),
    }
    dictionary = False
    if dictionary:
        losses = {
            "start": "categorical_crossentropy",
            "end": "categorical_crossentropy",
            "type": "categorical_crossentropy",
        }

        lossWeights = {"start": 1.0, "end": 1.0, "type": 1.0}

    else:
        losses = ["categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy"]
        lossWeights = [1.0, 1.0, 1.0]

    parser = argparse.ArgumentParser()

    # LARGE BERT
    # parser.add_argument("--model_config", default="input/transformers_cache/bert_large_uncased_config.json", type=str)
    # parser.add_argument("--checkpoint_dir", default="input/nq_bert_uncased_68", type=str)
    # parser.add_argument("--vocab_txt", default="input/transformers_cache/bert_large_uncased_vocab.txt", type=str)

    # Other parameters
    parser.add_argument('--short_null_score_diff_threshold', type=float, default=0.0)
    parser.add_argument('--long_null_score_diff_threshold', type=float, default=0.0)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--doc_stride", default=256, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--per_tpu_eval_batch_size", default=4, type=int)
    parser.add_argument("--n_best_size", default=10, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--verbose_logging", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--p_keep_impossible', type=float,
                        default=0.1, help="The fraction of impossible"
                                          " samples to keep.")
    parser.add_argument('--do_enumerate', action='store_true')

    args, _ = parser.parse_known_args()
    # assert args.model_type not in ('xlnet', 'xlm'), f'Unsupported model_type: {args.model_type}'

    do_lower_case = 'uncased'
    if namemodel == "bert":  # base
        model_config = 'input/transformers_cache/bert_base_uncased_config.json'
        vocab = 'input/transformers_cache/bert_base_uncased_vocab.txt'
    elif namemodel == 'albert':  # base v2
        model_config = 'input/transformers_cache/albert_base_v2.json'
        vocab = 'input/transformers_cache/albert-base-v2-spiece.model'
    elif namemodel == 'roberta':
        do_lower_case = False
        model_config = 'lo aggiungero in futuro'
        vocab = 'lo aggiungero in futuro'
    else:
        # di default metto il base bert
        model_config = 'input/transformers_cache/bert_base_uncased_config.json'
        vocab = 'input/transformers_cache/bert_base_uncased_vocab.txt'
        namemodel = "bert"
        print("sei impazzuto?")

    # Set XLA
    # https://github.com/kamalkraj/ALBERT-TF2.0/blob/8d0cc211361e81a648bf846d8ec84225273db0e4/run_classifer.py#L136
    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({'pin_to_host_optimization': False})

    print("Training / evaluation parameters %s", args)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[namemodel]
    config = config_class.from_json_file(model_config)

    print(model_class)
    mymodel = model_class(config)

    mymodel.compile(loss=losses, loss_weights=lossWeights)

    x, y = dataset_utils.getTokenizedDataset(namemodel, vocab, do_lower_case, namefile, verbose, evaluate,
                                             max_num_samples)

    print("FITTING")

    cb = TimingCallback()  # Callback per il tempo di esecuzione
    mymodel.fit(x, y, verbose=1, batch_size=batch_size, epochs=1, callbacks=[cb, tboard_callback])
    mymodel.summary()
    print("Time: " + str(cb.logs))


if __name__ == "__main__":
    main('bert', 4, 'prova.jsonl', evaluate=False, verbose=False)
