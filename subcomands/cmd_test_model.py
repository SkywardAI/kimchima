import sys

import torch
from subcomands import SubCommandLineInterface
from tokenizer import Tokenizer

from tqdm import tqdm



class CommandTestModel(SubCommandLineInterface):

    @staticmethod
    def load_model(model_dir: str, tokenizer_path: str):
        return super(CommandTestModel,CommandTestModel).load_model(model_dir=model_dir, tokenizer_path=tokenizer_path)
    
    @staticmethod
    def t_model(args):
        try:
            model_dir=args.model_dir
            tokenizer_path=args.tokenizer_path

            with torch.no_grad():
                tok=Tokenizer(model_path=tokenizer_path)
                llama=CommandTestModel.load_model(model_dir, tokenizer_path)

                tokens = tok.encode("Hello, I am", True, False)
                for i in tqdm(range(0,10)):
                    token_tensor = torch.tensor(tokens)
                    logits = llama(token_tensor.unsqueeze(0), 0)
                    sample = logits[:, -1,:].argmax(dim=-1).item()
                    # print(f'Sample is {sample} {tok.decode(sample)}')
                    tokens=tokens+[sample]
            decode=tok.decode(tokens)
            print(f'Sampled output is {decode}')
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)