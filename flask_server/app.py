from flask import Flask, jsonify, request, abort
import time
import answerBot

debug = False
app = Flask(__name__)
mybot = answerBot.AnswerBot()


@app.route("/answer",  methods=["POST"])
def answer():
    question = dict(request.form)['query']
    if debug:
        print("i recived this question: {}".format(question))

    # Errors
    if not question:
        return abort(400) # BAD REQUEST
     # TODO implement question analisys and answer

    answer = mybot.answer(question)

    ret = {"answer":answer, "question":question}
    return jsonify(ret)