from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
from sys import platform
from dataset_utils_version2 import *
from model_stuff.TFAlbertForNaturalQuestionAnswering import TFAlbertForNaturalQuestionAnswering
from model_stuff.TFBertForNaturalQuestionAnswering import TFBertForNaturalQuestionAnswering


def main(namemodel, args, checkpoint, namefile, verbose=False, max_num_samples=1_000_000, do_cache=False):
    """
    :param do_cache:
    :param namefile:
    :param args:
    :param checkpoint:
    :param namemodel: nomde del modello da eseguire
    :param verbose: fag per stampare informazioni sul primo elemento del datasetk
    :param max_num_samples: massimo numero di oggetti da prendere in considerazione (1mil Default)
    :return: TUTTO

    """

    if platform == "win32":
        logs = "logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")  # Windows
    else:
        logs = "log/" + datetime.now().strftime("%Y%m%d-%H%M%S")  # Linux

    if not os.path.exists(logs):
        os.makedirs(logs)

    MODEL_CLASSES = {
        'bert': (BertConfig, TFBertForNaturalQuestionAnswering, BertTokenizer),
        'bert_large': (BertConfig, TFBertForNaturalQuestionAnswering, BertTokenizer),
        'albert': (AlbertConfig, TFAlbertForNaturalQuestionAnswering, AlbertTokenizer),  # V2
    }

    if namemodel == "bert":  # base
        pretrained = 'bert-base-uncased'
        vocab = 'bert-base-uncased'

    elif namemodel == 'albert':  # base v2
        pretrained = 'albert-base-v2'
        vocab = 'albert-base-v2'

    elif namemodel == "bert_large":
        vocab = 'bert-base-uncased'
        pretrained = 'bert-large-uncased'

    else:
        pretrained = 'bert-base-uncased'
        vocab = 'bert-base-uncased'

    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({'pin_to_host_optimization': False})

    config_class, model_class, tokenizer_class = MODEL_CLASSES[namemodel]
    config = config_class.from_pretrained(pretrained)
    print(tokenizer_class)

    # load tokenizer from pretrained
    tokenizer = tokenizer_class.from_pretrained(vocab)

    mymodel = model_class(config)
    mymodel(mymodel.dummy_inputs, training=False)
    mymodel.load_weights(os.path.join(checkpoint, "weights.h5"), by_name=True)
    print("Checkpoint loaded succefully")

    if namemodel == 'bert':
        tags = get_add_tokens(do_enumerate=args.do_enumerate)
        num_added = tokenizer.add_tokens(tags)
        print(f"Added {num_added} tokens")

    if namefile != '':
        print("***** Running evaluation *****")
        eval_ds, crops, entries, eval_dataset_length = getDatasetForEvaluation(args, tokenizer, namefile, verbose,
                                                                               max_num_samples, do_cache)
        print("***** Getting results *****")
        getResult(args, mymodel, eval_ds, crops, entries, eval_dataset_length, do_cache, namefile, tokenizer)
    else:
        return mymodel, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Other parameters
    parser.add_argument('--short_null_score_diff_threshold', type=float, default=0.0)
    parser.add_argument('--long_null_score_diff_threshold', type=float, default=0.0)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--doc_stride", default=256, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--per_tpu_eval_batch_size", default=4, type=int)
    parser.add_argument("--n_best_size", default=5, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--verbose_logging", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--p_keep_impossible', type=float,
                        default=0.1, help="The fraction of impossible"
                                          " samples to keep.")
    parser.add_argument('--do_enumerate', action='store_true')

    parser.add_argument("--checkpoint", default="../checkpoints/BERTWITHTOKEN2EPOCHSCHP/checkpoint-194000", type=str,
                        help="The file we will use as checkpoint")

    parser.add_argument('--test_dir', type=str, default='../TestData/simplified-nq-test.jsonl',
                        help='Path of test set')

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--do_cache', type=bool, default=True)
    parser.add_argument('--true_endToken', type=bool, default=True, help="Approach for end token, True for computed")
    parser.add_argument('--eval_method', type=str, default='', help='''
                        method:
                            1) ''          = default
                            2) 'restoring' = if rejected and short IN long text ->taken (text)
                            3) 'matching'  = taking the best short IN long token (token)
                            4) 'mixed'     = 2. and 3. mixed
    
                        ''')

    args, _ = parser.parse_known_args()
    print("Evaluation parameters ", args)

    main(args.model, args, args.checkpoint, args.test_dir, verbose=args.verbose, do_cache=args.do_cache)
