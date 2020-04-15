from flask import Flask, jsonify, request, abort
#Rimport answerBot
#from config import config

debug = False
app = Flask(__name__)
#mybot = answerBot.AnswerBot(config.path)


@app.route("/answer",  methods=["POST"])
def answer():
    query = dict(request.form)['query']
    if debug:
        print("i recived this query: {}".format(query))

    # Errors
    if not query:
        return abort(400) # BAD REQUEST
     # TODO implement query analisys and answer

    ret = {"data":query}
    return jsonify(ret)