import tensorflow as tf


class AnswerBot:
    def __init__(self, model_path):
        super().__init__()
        # load the model
        self.model = tf.keras.loadmodel(model_path)

    def answer(self, question):
        page = self.obtain_wiki_page(question)
        page = self.preprocesspage(page)
        prediction = self.model(page)
        return self.frompredictiontotext(prediction, page)