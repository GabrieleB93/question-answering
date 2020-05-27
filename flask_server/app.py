import argparse
import answerBot
from flask import Flask, jsonify, request, abort
from flask_cors import CORS

debug = False
app = Flask(__name__)
CORS(app)


@app.route("/answer", methods=["POST"])
def answer():
    question = dict(request.form)['query']
    print(question)
    if debug:
        print("i recived this question: {}".format(question))

    # Errors
    if not question:
        return abort(400)  # BAD REQUEST
    # TODO implement question analisys and answer

    answer = mybot.answer(question)
    print(answer)
    for ex, pred in answer.items():
        answer_short, start_short, end_short, answer_long, start_long = pred
        end_long = start_long + len(answer_long)

    if answer_long == '' and answer_short == '':
        answer_long = 'Not answer found'
    elif answer_long == '' and not answer_short == '':
        answer_long = answer_short

    ret = {"answer_long": answer_long, "answer_short": answer_short, "start_short": start_short,
           "end_short": end_short, "start_long": start_long, "end_long": end_long}
    print(ret)
    return jsonify(ret)


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
    parser.add_argument("--max_answer_length", default=40, type=int)
    parser.add_argument("--verbose_logging", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--p_keep_impossible', type=float,
                        default=0.1, help="The fraction of impossible"
                                          " samples to keep.")
    parser.add_argument('--do_enumerate', action='store_true')

    parser.add_argument("--checkpoint", default="../checkpoints/BERTWITHTOKEN2EPOCHSCHP/checkpoint-194000", type=str,
                        help="The file we will use as checkpoint")

    parser.add_argument('--test_dir', type=str, default='TestData/simplified-nq-test.jsonl',
                        help='Directory were all the traing data splitted in smaller junks are stored')

    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--eval_method', type=str, default='')


    args, _ = parser.parse_known_args()

    mybot = answerBot.AnswerBot(args.batch_size, args.model, args.checkpoint, args, verbose=args.verbose)
    app.run()
