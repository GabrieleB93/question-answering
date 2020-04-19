import tensorflow as tf
from web_to_simplified import create_simplified_input
from query_to_page import query_to_page

class AnswerBot:
    def __init__(self, verbose = False):
        self.verbose = verbose

    """def __init__(self, model_path):
        super().__init__()
        # load the model
        self.model = tf.keras.loadmodel(model_path)"""

    def preprocess_page(self, page, name):
        '''
        This function takes a page as input and returns a
        simplified_nq_example

        @param page the page containing the abstract

        @retval the simplified_nq_example
        '''

        # TODO remove [edit] and remove referneces [1]
        return create_simplified_input(page.query, page.url, page.content, name)

    def obtain_wiki_page(self, q):
        """This function returns a Page object from the question q

        @param q query string

        @retval the page object containing query, url and html content

        """
        return query_to_page(q)

    def answer(self, question, name = None):
        page = self.obtain_wiki_page(question)
        simplified_datum = self.preprocess_page(page, name)
        if self.verbose:
            print('we obtained this page (url): ', page)
            print('this is the page text processed: ', simplified_datum)
        return simplified_datum
        #prediction = self.model(page)
        #return self.frompredictiontotext(prediction, page)

if __name__=="__name__":
    question = "where was shrodinger born?"
    a = AnswerBot()
    page = a.obtain_wiki_page(question)
    simplified_datum = a.preprocess_page(page)
    print(simplified_datum)
