import unittest

from distillbert import load_modal_and_tokenizer, predict


class Testcase(unittest.TestCase):

    def test_predict(self):
        model, tokenizer = load_modal_and_tokenizer()
        prediction = predict("this is a fake product", model, tokenizer)
        print(prediction)
        self.assertTrue(prediction)