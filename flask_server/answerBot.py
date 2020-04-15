import tensorflow as tf

from web_to_simplified import create_simplified_input

class Page:
    query
    url
    content

    def __init__(self, q, u, c):
        self.query = q
        self.url = u
        self.content = c


class AnswerBot:
    def __init__(self, model_path):
        super().__init__()
        # load the model
        self.model = tf.keras.loadmodel(model_path)

    def preprocess_page(page):
    """
        This function takes a page as input and returns a
        simplified_nq_example

        @param page the page containing the abstract

        @retval the simplified_nq_example
    """
        return create_simplified_input(page.query, page.url, page.content)

    def answer(self, question):
        page = self.obtain_wiki_page(question)
        simplified_datum = self.preprocess_page(page)
        prediction = self.model(page)
        return self.frompredictiontotext(prediction, page)