# coding=utf-8

import unittest

from pkg.auto import Auto


class TestAuto(unittest.TestCase):

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    def test_get_embeddings(self):
        model = Auto(model_name_or_path=self.model_name)
        embeddings = model.get_embeddings(text='Melbourne')
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape, (1, 384))

    def test_get_embeddings_with_list(self):
        model = Auto(model_name_or_path=self.model_name)
        embeddings = model.get_embeddings(text=['Melbourne', 'Sydney'])
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape, (2, 384))
