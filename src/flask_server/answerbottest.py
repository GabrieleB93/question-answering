import answerBot
import json

queries = ["who discovered antartic", "When did napoleon born?"]
if __name__ == "__main__":
    bot = answerBot.AnswerBot(verbose= True)
    
    result = {}
    for query in queries:
        result[query] = bot.answer(query, query[:3])

    with open('query.json', 'w') as outfile:
        json.dump(result, outfile)



    