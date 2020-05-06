
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
from datetime import datetime
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import BertConfig, BertTokenizer, AlbertTokenizer, AlbertConfig, AutoTokenizer

from dataset_utils import getTokenizedDataset
from generator import DataGenerator
from model_stuff import model_utils as mu
from model_stuff.TFAlbertForNaturalQuestionAnswering import TFAlbertForNaturalQuestionAnswering
from model_stuff.TFBertForNaturalQuestionAnswering import TFBertForNaturalQuestionAnswering


def main(namemodel, batch_size, train_dir, val_dir, epoch, checkpoint_dir, do_cache=False, verbose=False, evaluate=False,
         max_num_samples=1_000_000, checkpoint="", log_dir="log/", learning_rate=0.005, starting_epoch = 0):
    """

    :param do_cache:
    :param learning_rate:
    :param log_dir:
    :param checkpoint:
    :param epoch:
    :param train_dir:
    :param val_dir:
    :param checkpoint_dir:
    :param namemodel: nomde del modello da eseguire
    :param batch_size: dimensione del batch durante il training
    :param verbose: fag per stampare informazioni sul primo elemento del dataset
    :param evaluate: Bool per indicare se dobbiamo eseguire Evaluation o Training. Training di Default
    :param max_num_samples: massimo numero di oggetti da prendere in considerazione (1mil Default)
    :return: TUTTO

    """
    logs = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))  # Linux
    # logs = "logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")  # Windows
    if not os.path.exists(logs):
        os.makedirs(logs)
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                     histogram_freq=1,
                                                     update_freq='batch',
                                                     profile_batch=0)

    MODEL_CLASSES = {
        'bert': (BertConfig, TFBertForNaturalQuestionAnswering, BertTokenizer),
        'albert': (AlbertConfig, TFAlbertForNaturalQuestionAnswering, AlbertTokenizer),  # V2
        'albert_squad': (AlbertConfig, TFAlbertForNaturalQuestionAnswering,
                         AutoTokenizer.from_pretrained("twmkn9/albert-base-v2-squad2"))
        # 'roberta': (RobertaConfig, TFRobertaForNaturalQuestionAnswering, RobertaTokenizer),
    }

    # define the losses. We decided to use the Sparse one because our targert are integer and not one hot vector
    # and from logit because we don't apply the softmax
    losses = [tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), ]
    lossWeights = [1.0, 0.5, 0.5]

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

    elif namemodel == "albert_squad":
        model_config = 'input/transformers_cache/albert_base_v2_squad.json'
        vocab = 'input/transformers_cache/albert-base-v2-spiece.model'
        tokenizer = AutoTokenizer.from_pretrained("twmkn9/albert-base-v2-squad2")

    else:
        # di default metto il base albert
        model_config = 'input/transformers_cache/albert_base_v2.json'
        vocab = 'input/transformers_cache/albert-base-v2-spiece.model'
        namemodel = "albert"

    # Set XLA
    # https://github.com/kamalkraj/ALBERT-TF2.0/blob/8d0cc211361e81a648bf846_d8ec84225273db0e4/run_classifer.py#L136
    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({'pin_to_host_optimization': False})

    config_class, model_class, tokenizer_class = MODEL_CLASSES[namemodel]
    config = config_class.from_json_file(model_config)

    print(model_class)
    mymodel = model_class(config)

    if checkpoint != "":
        # we do this in order to compile the model, otherwise it will not be able to lead the weights
        # mymodel(traingenerator.get_sample_data())
        mymodel(mymodel.dummy_inputs)
        if starting_epoch == 0:
            startepoch = os.path.split(checkpoint)[-1]
            startepoch = re.sub('weights.', '', startepoch)
            startepoch = int(startepoch.split("-")[0])
            initial_epoch = startepoch
        else:
            initial_epoch = starting_epoch

        mymodel.load_weights(checkpoint, by_name=True)
        print("checkpoint loaded succefully")
    else:
        startepoch = None
        initial_epoch = 0

    adam = tfa.optimizers.AdamW(lr=learning_rate, weight_decay=0.01, epsilon = 1e-6)

    mymodel.compile(loss=losses,
                    loss_weights=lossWeights,
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='Accuracy')],
                    optimizer=adam
                    )


    config = AlbertConfig.from_pretrained('input/transformers_cache/albert_base_v2.json')
    mymodel = TFAlbertForNaturalQuestionAnswering.from_pretrained('albert-base-v2', from_pt=True, config=config)

    x, y = getTokenizedDataset(namemodel, vocab, 'uncased', "validationData/2.jsonl", verbose, max_num_samples)
    # mymodel.evaluate(x, y, batch_size)
    outputs = mymodel(x, training=False)
    tf.print(outputs)
    print(tf.argmax(outputs[0][0][1:]))
    print(tf.argmax(outputs[1][0][1:]))
    print(tf.argmax(outputs[2][0][1:]))
    mymodel.summary()

if __name__ == "__main__":
    #tf.config.gpu.set_per_process_memory_fraction(0.50)
    #tf.config.gpu.set_per_process_memory_growth(True)

    parser = argparse.ArgumentParser()

    # Other parameters
    parser.add_argument("--learning_rate", default=5e-3, type=float, )

    parser.add_argument("--checkpoint_dir", default="checkpoints/", type=str,
                        help="the directory where we want to save the checkpoint")
    parser.add_argument("--checkpoint", default="", type=str, help="The file we will use as checkpoint")

    parser.add_argument('--validation_dir', type=str, default='validationData/',
                        help='Directory where all the validation data splitted in smaller junks are stored')
    parser.add_argument('--train_dir', type=str, default='TrainData/',
                        help='Directory where all the traing data splitted in smaller junks are stored')

    parser.add_argument('--log_dir', type=str, default='log/',
                        help='Directory for tensorboard')
    # Quelli sopra andrebbero tolti perchè preenti anche dentro dataset_utils, di conseguenza andrebbero passati
    # tramite generator -> to fix
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='albert')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--do_cache', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--starting_epoch', type=int, default=0)

    args, _ = parser.parse_known_args()
    # assert args.model_type not in ('xlnet', 'xlm'), f'Unsupported model_type: {args.model_type}'
    print("Training / evaluation parameters %s", args)

    main(args.model, args.batch_size, args.train_dir, args.validation_dir, args.epoch, args.checkpoint_dir,
         checkpoint=args.checkpoint,do_cache=args.do_cache,
         evaluate=args.evaluate, verbose=args.verbose, log_dir=args.log_dir, learning_rate=args.learning_rate, starting_epoch = args.starting_epoch)
