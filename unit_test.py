import unittest
from distillbert import *
class TestDistill(unittest.TestCase):
    def test_loading_model(self):
        model = train(2e-5,10, 10)
        self.assertTrue(model)

    def test_load_dataset(self):
        dataset = load_dataset()
        self.assertIn("train", dataset)