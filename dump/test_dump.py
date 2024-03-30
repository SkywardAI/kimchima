import unittest
import torch
from model import Transformer, ModelArgs

from dump import Dump

class TestDump(unittest.TestCase):

    @unittest.skip("Pass testing for no useful features")
    def test_dump(self):
        n_vocab = 10
        n_state = 8
        multiple_of = 3
        n_head = 4
        n_kv_head = 2
        n_layer = 3
        norm_eps = 1e-6
        max_batch_size = 1


        model_args = ModelArgs(
            dim=n_state,
            n_layers=n_layer,
            n_heads=n_head,
            n_kv_heads=n_kv_head,
            vocab_size=n_vocab,
            multiple_of=multiple_of,
            norm_eps=norm_eps,
            max_batch_size=max_batch_size,
        )

        llama = Transformer(model_args)

        with torch.no_grad():
            tokens=torch.tensor([0,2,1], dtype=torch.int32).unsqueeze(0)
            output=llama(tokens,0)
            self.assertIsNotNone(tokens.numpy())
            self.assertIsNotNone(output.numpy())
        
            Dump.save_transformer(llama, 'params')
