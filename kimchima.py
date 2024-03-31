import argparse

from cmds.auto import CommandAuto


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-command help")

    parser_command_auto=subparsers.add_parser("auto", help="auto help")
    parser_command_auto.add_argument("model_name_or_path", help="model name or path")
    parser_command_auto.add_argument("text", help="text str or list of text str")
    parser_command_auto.set_defaults(func=CommandAuto.auto)

    # parser_command_test=subparsers.add_parser("test", help="test help")
    # parser_command_test.add_argument("model_dir", help="model directory")
    # parser_command_test.add_argument("tokenizer_path", help="tokenizer path")
    # parser_command_test.set_defaults(func=CommandTestModel.t_model)

    # parser_command_dump=subparsers.add_parser("dump", help="dump help")
    # parser_command_dump.add_argument("model_dir", help="model directory")
    # parser_command_dump.add_argument("tokenizer_path", help="tokenizer path")
    # parser_command_dump.set_defaults(func=CommandDumpModel.dump_model)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()