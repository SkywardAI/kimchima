import sys
from subcomands import SubCommandLineInterface
from dump import Dump


class CommandDumpModel(SubCommandLineInterface):
    
        @staticmethod
        def load_model(model_dir: str, tokenizer_path: str):
            return super(CommandDumpModel,CommandDumpModel).load_model(model_dir=model_dir, tokenizer_path=tokenizer_path)
        
        @staticmethod
        def dump_model(args):
            try:
                model=CommandDumpModel.load_model(args.model_dir, args.tokenizer_path)
                Dump.save_transformer(model, 'params')
            except Exception as e:
                print(f"An error occurred: {e}")
                sys.exit(1)