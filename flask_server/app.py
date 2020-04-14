from flask import Flask, jsonify, request, abort
import answerBot
from config import config

app = Flask(__name__)
mybot = answerBot.AnswerBot(config.path)

@app.route("/") 
def hello_world():
    return "Hello, World!"

@app.route("/answer",  methods=["POST"])
def answer():
    req = request.get_json()
    req['keywordExpansion'] = []
    
    # Errors
    if not req:
        return abort(400) # BAD REQUEST
    if 'userProfile' not in req:
        req['adaptionError'] = {"userProfile not found"}
        return jsonify(req), 400
    if 'tastes' not in req['userProfile']:
        req['adaptionError'] = {"userProfile incomplete"}
		
	print("query ricevuta")
    
    return jsonify(req)