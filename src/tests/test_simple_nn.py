import unittest
from models.simple_nn import SimpleNN


class TestSimpleNN(unittest.TestCase):
    def test_backward_pass(self):
        self.assertTrue(SimpleNN.backward_pass())
