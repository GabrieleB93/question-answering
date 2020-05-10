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
from generator import DataGenerator
from model_stuff import model_utils as mu
from model_stuff.TFAlbertForNaturalQuestionAnswering import TFAlbertForNaturalQuestionAnswering
from model_stuff.TFBertForNaturalQuestionAnswering import TFBertForNaturalQuestionAnswering
import dataset_utils_version2 as dataset_utils
from tqdm import tqdm; tqdm.monitor_interval = 0  #
import glob
import logging
from shutil import rmtree, copy

logger = logging.getLogger(__name__)


def main(namemodel, 
        batch_size, 
        train_dir, 
        epoch, checkpoint_dir, do_cache=False, verbose=False,
        evaluate=False,
        max_num_samples=1_000_000, 
        checkpoint="", 
        log_dir="log/", 
        learning_rate=0.005, 
        starting_epoch=0,
        checkpoint_interval = 1000):
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


    MODEL_CLASSES = {
        'bert': (BertConfig, TFBertForNaturalQuestionAnswering, BertTokenizer),
        'albert': (AlbertConfig, TFAlbertForNaturalQuestionAnswering, AlbertTokenizer),  # V2
        'albert_squad': (AlbertConfig, TFAlbertForNaturalQuestionAnswering,
                         AutoTokenizer.from_pretrained("twmkn9/albert-base-v2-squad2"))
        # 'roberta': (RobertaConfig, TFRobertaForNaturalQuestionAnswering, RobertaTokenizer),
    }

    do_lower_case = 'uncased'

    if namemodel == "bert":  # base
        model_config = 'input/transformers_cache/bert_base_uncased_config.json'
        vocab = 'input/transformers_cache/bert_base_uncased_vocab.txt'
        pretrained = ''

    elif namemodel == 'albert':  # base v2
        model_config = 'input/transformers_cache/albert_base_v2.json'
        vocab = 'input/transformers_cache/albert-base-v2-spiece.model'
        pretrained = 'albert-base-v2'


    elif namemodel == 'roberta':
        do_lower_case = False
        model_config = 'lo aggiungero in futuro'
        vocab = 'lo aggiungero in futuro'
        pretrained = 'roberta-'

    elif namemodel == "albert_squad":
        model_config = 'input/transformers_cache/albert_base_v2_squad.json'
        vocab = 'input/transformers_cache/albert-base-v2-spiece.model'
        tokenizer = AutoTokenizer.from_pretrained("twmkn9/albert-base-v2-squad2")
        pretrained = ''

    else:
        # di default metto il base albert
        model_config = 'input/transformers_cache/albert_base_v2.json'
        vocab = 'input/transformers_cache/albert-base-v2-spiece.model'
        namemodel = "albert"
        pretrained = 'albert-base-v2'

    # Set XLA
    # https://github.com/kamalkraj/ALBERT-TF2.0/blob/8d0cc211361e81a648bf846_d8ec84225273db0e4/run_classifer.py#L136
    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({'pin_to_host_optimization': False})


    config_class, model_class, tokenizer_class = MODEL_CLASSES[namemodel]
    
    tokenizer = tokenizer_class.from_pretrained(pretrained,
        do_lower_case=do_lower_case)

    """
    tags = dataset_utils.get_add_tokens(do_enumerate=True)
    num_added = tokenizer.add_tokens(tags)
    print(f"Added {num_added} tokens")
    """
    print(model_class)
    start_file = 0
    if checkpoint != "":
        config = config_class.from_json_file(model_config)
        mymodel = model_class(config)

        # we do this in order to compile the model, otherwise it will not be able to lead the weights
        # mymodel(traingenerator.get_sample_data())
        
        mymodel(mymodel.dummy_inputs)
        if starting_epoch == 0:
            start_file = os.path.split(checkpoint)[-1]
            start_file = re.sub('weights.', '', start_file)
            start_file = int(start_file.split("-")[0])
            start_file = start_file
        else:
            start_file = starting_epoch

        mymodel.load_weights(checkpoint, by_name=True)
        print("checkpoint loaded succefully")
    else:
        initial_epoch = 0
        config = config_class.from_pretrained(model_config)
        #mymodel = model_class(config, training = True)
        mymodel = model_class.from_pretrained(pretrained, config=config)

    adam = tfa.optimizers.AdamW(lr=learning_rate, weight_decay=0.01, epsilon=1e-6)


        # this file we implement the training by ourself istead of using keras
    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            input_ley =  ['input_ids','attention_mask', 'token_type_ids']
            outputs = mymodel({ k: batch[k] for k in input_ley } , training=True)
            
            # (batch_size, 1).* type_answer -> (batch_size, 1)

            type_loss = tf.keras.losses.binary_crossentropy(batch["answerable"], outputs["answerable"])

            start_loss = tf.keras.losses.sparse_categorical_crossentropy(batch["start"], outputs["start"], from_logits=True)
            end_loss = tf.keras.losses.sparse_categorical_crossentropy(batch["end"], outputs["end"], from_logits=True)
            long_loss = tf.keras.losses.sparse_categorical_crossentropy(batch["long"], outputs["long"], from_logits=True)
            
            # Idea: if not answerable we not optimize for the start, end long losses



            acc_1 = tf.keras.metrics.sparse_categorical_accuracy(
                batch["start"], outputs["start"])          
            acc_2 = tf.keras.metrics.sparse_categorical_accuracy(
                batch["end"], outputs["end"]) 
            acc_3 = tf.keras.metrics.sparse_categorical_accuracy(
                batch["long"], outputs["long"])  

            loss = ((tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss) / 2.0) +
                tf.reduce_mean(long_loss) ) / 2.0 + tf.reduce_mean(type_loss) / 2.0

        type_loss = tf.keras.metrics.binary_accuracy(tf.cast(batch["answerable"], float), outputs["answerable"], threshold=0.5) 

        start_loss = tf.math.multiply(start_loss,tf.cast(batch["answerable"], float))
        end_loss = tf.math.multiply(end_loss, tf.cast(batch["answerable"], float))
        long_loss = tf.math.multiply(long_loss, tf.cast(batch["answerable"], float))

        grads = tape.gradient(loss, mymodel.trainable_variables)
        adam.apply_gradients(zip(grads, mymodel.trainable_variables))
        return loss, tf.reduce_mean(acc_1), tf.reduce_mean(acc_2), tf.reduce_mean(acc_3), tf.reduce_mean(type_loss)

    all_files = os.listdir(train_dir)  # list of all the files from the directory
    all_files = sorted(all_files, key=lambda file1: int(file1[:-6]))
    allfFile_copy = all_files.copy()

    if start_file > 0:
        print("since we loaded drom a checkpoints we had this files: ",all_files, start_file)
        all_files = all_files[start_file:]
        print("since we loaded drom a checkpoints we have this files: ",all_files, start_file)
    

    global_step = 1
    num_samples = 0
    smooth = 0.99
    smooth_acc = 0.66
    running_loss = 0.0
    running_accuracy_1 = 0.0
    running_accuracy_2 = 0.0
    running_accuracy_3 = 0.0
    running_type_loss = 0.0
    for i, file in enumerate(all_files):
        # load file 
        train_dataset = dataset_utils.getTokenizedDataset(  tokenizer,
                                                            os.path.join(train_dir, file),
                                                            verbose,
                                                            max_num_samples)

        # how many epochs iterations we do in this file
        num_steps_per_epoch = len(train_dataset['input_ids']) // batch_size

        opt = tf.data.Options()
        opt.experimental_deterministic = True
        # use tf.Data in order to create an efficent pipeLine
        train_ds = tf.data.Dataset.from_tensor_slices(train_dataset).with_options(opt)
        train_ds = train_ds.repeat()
        train_ds = train_ds.shuffle(buffer_size=100, seed=12)
        train_ds = train_ds.batch(batch_size=batch_size, drop_remainder=True)
        train_ds = iter(train_ds)

        epoch_iterator = tqdm(range(num_steps_per_epoch))
        for step in epoch_iterator:
            batch = next(train_ds)
            loss, accuracy_1, accuracy_2, accuracy_3, type_loss = train_step(batch)
        
            global_step += 1
            num_samples += batch_size
            running_loss = smooth * running_loss + (1. - smooth) * float(loss)
            running_accuracy_1 =  smooth_acc * running_accuracy_1 + (1. - smooth_acc) * float(accuracy_1)
            running_accuracy_2 =  smooth_acc * running_accuracy_2 + (1. - smooth_acc) * float(accuracy_2)
            running_accuracy_3 =  smooth_acc * running_accuracy_3 + (1. - smooth_acc) * float(accuracy_3)
            running_type_loss = smooth_acc * running_type_loss + (1. - smooth_acc) * float(type_loss)

            if global_step % checkpoint_interval == 0:
                # Save model checkpoint
                step_str = '%06d' % global_step
                ckpt_dir = os.path.join(checkpoint_dir, 'checkpoint-{}'.format(step_str))
                os.makedirs(ckpt_dir, exist_ok=True)
                weights_fn = os.path.join(ckpt_dir, 'weights.h5')
                mymodel.save_weights(weights_fn)
                tokenizer.save_pretrained(ckpt_dir)

                # remove too many checkpoints
                checkpoint_fns = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint-*')))
                for fn in checkpoint_fns[:-2]:
                    rmtree(fn)
                
            
            epoch_iterator.set_postfix({'file': '%d/%d' % (i, len(all_files)),
                    'samples': num_samples, 'global_loss': round(running_loss, 4), 
                    "Accuracy": "%.2f:%.2f:%.2f:%.2f" % (running_accuracy_1, running_accuracy_2, running_accuracy_3, running_type_loss) })





if __name__ == "__main__":
    # tf.config.gpu.set_per_process_memory_fraction(0.50)
    # tf.config.gpu.set_per_process_memory_growth(True)

    parser = argparse.ArgumentParser()

    # Other parameters
    parser.add_argument("--learning_rate", default=5e-3, type=float, )

    parser.add_argument("--checkpoint_dir", default="checkpoints/", type=str,
                        help="the directory where we want to save the checkpoint")
    parser.add_argument("--checkpoint", default="", type=str, help="The file we will use as checkpoint")

    parser.add_argument('--train_dir', type=str, default='TrainData/',
                        help='Directory where all the traing data splitted in smaller junks are stored')

    parser.add_argument('--log_dir', type=str, default='log/',
                        help='Directory for tensorboard')
    
    parser.add_argument("--checkpoint_interval", type = int, default=1000,
            help="after how many steps do we have to save the checkpoint")

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

    main(args.model, 
        args.batch_size, 
        args.train_dir, 
        args.epoch, 
        args.checkpoint_dir,
        checkpoint=args.checkpoint, 
        do_cache=args.do_cache,
        evaluate=args.evaluate, 
        verbose=args.verbose, 
        log_dir=args.log_dir, 
        learning_rate=args.learning_rate,
        starting_epoch=args.starting_epoch,
        checkpoint_interval = args.checkpoint_interval)
