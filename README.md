# kimchi

The collections of tools for testing and dumping LLMs. And this project is inspired by [Llama2-burn Project](https://github.com/Gadersd/llama2-burn/tree/main). And the main purpose of this project is to make sure the Llama2 model works well before we load it into the Rust ML framework.


# Usage

You can use it as a command line tool if you like. And you can also use it as a library. Or you can run it in VSCode with [`launch.json`](.vscode/launch.json).



### Test the model

```bash
kimchi test <model_dir> <tokenizer_path>
# or
python3 kimchi.py test <model_dir> <tokenizer_path>
```

### Dump the model

```bash
kimchi dump <model_dir> <tokenizer_path>
# or
python3 kimchi.py dump <model_dir> <tokenizer_path>
```

```
pip install torch==2.2.1
pip install sentencepiece==0.2.0
pip install transformers==4.39.1
```

# Credits
- [Llama2-burn Project](https://github.com/Gadersd/llama2-burn/tree/main)


# License
This project is licensed as specified in the [LICENSE](./LICENSE) file.