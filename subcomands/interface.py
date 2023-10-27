from __future__ import annotations

from pkg import Devices
import json

from tokenizer import Tokenizer
from model import Transformer, ModelArgs
from pathlib import Path
import torch

class SubCommandLineInterface:

    @staticmethod
    def load_model(model_dir:str, tokenizer_path:str):
        
        tok = Tokenizer(model_path=tokenizer_path)
        checkpoints = sorted(Path(model_dir).glob("*.pth"))
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoints found in {model_dir}")
        #TODO: check the load to mps may be got error, it does not support f16
        weights = [torch.load(filename, map_location=Devices.CPU.value) for filename in checkpoints]
        with open(Path(model_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        
        model_args: ModelArgs = ModelArgs(
            max_batch_size=1,
            **params,
        )

        model_args.vocab_size = tok.n_words
        model = Transformer(model_args)
        model.load_state_dict(SubCommandLineInterface._concat_weights(weights), strict=False)
        model.max_seq_len = model.tok_embeddings.weight.shape[0]
        return model
    
    @staticmethod
    # The concat_weights function is adapted from the tinygrad library:  
    # https://github.com/tinygrad/tinygrad/blob/master/tinygrad/examples/llama.py
    # Original code by TinyGrad authors
    # Adapted by [Aisuko]
    def _concat_weights(models):
        def convert(name) -> torch.Tensor:
            disk_tensors =[model[name] for model in models]
            if len(disk_tensors) ==1 or len(disk_tensors[0].shape) ==1:
                return disk_tensors[0]
            axis = 1 if name.startwith('tok_embeddings.') or name.endswith('.attention.wo.weight') or name.endswith('.feed_forward.w2.weight') else 0
            return disk_tensors[0].cat(*disk_tensors[1:], dim=axis)
        return {name: convert(name) for name in {name: None for model in models for name in model}}