from web_to_simplified import create_simplified_input
from query_to_page import query_to_page
import model_evaluation
from dataset_utils_version2 import *


class AnswerBot:
    def __init__(self, batch, namemodel, checkpoint, args, verbose=False):
        self.verbose = verbose
        self.bacth = batch  # included in args
        self.namemodel = namemodel
        self.checkpoint = checkpoint
        self.args = args
        self.model, self.tokenizer = model_evaluation.main(namemodel, args, checkpoint, '')

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

    def answer(self, question, name=None):
        tmp_file = 'query.jsonl'
        page = self.obtain_wiki_page(question)
        simplified_datum = self.preprocess_page(page, name)
        if self.verbose:
            print('we obtained this page (url): ', page)
            print('this is the page text processed: ', simplified_datum)

        with open(tmp_file, 'w') as outfile:
            json.dump(simplified_datum, outfile)

        eval_ds, crops, entries, eval_dataset_length = getDatasetForEvaluation(self.args, self.tokenizer, tmp_file,
                                                                               self.verbose, 1_000_000, False)
        print("***** Getting results *****")
        result = getResult(self.args, self.model, eval_ds, crops, entries, eval_dataset_length, False, tmp_file,
                           self.tokenizer, app=True)
        print(result)
        return result


if __name__ == "__name__":
    question = "where was shrodinger born?"
    a = AnswerBot()
    page = a.obtain_wiki_page(question)
    simplified_datum = a.preprocess_page(page)
    print(simplified_datum)
