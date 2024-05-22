# coding=utf-8
# Copyright (c) 2023 Aisuko

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from kimchima.pkg import ModelFactory, CrossEncoderFactory


class CommandAutoModel:
    """
    A class for loading models.
    """

    @staticmethod
    def auto(args):
        """
        Get embeddings of text.

        Args:
            args (argparse.Namespace): The arguments.

        Returns:
            torch.tensor: The embeddings of text.
        """
        model = ModelFactory.auto_model(pretrained_model_name_or_path=args.model_name_or_path)
        print(model.config)

class CommandCrossEncoder:
    """
    A class for loading models.
    """

    @staticmethod
    def auto(args):
        """
        Get embeddings of text.

        Args:
            args (argparse.Namespace): The arguments.

        Returns:
            torch.tensor: The embeddings of text.
        """
        query = "A man is eating pasta."

        # With all sentences in the corpus
        corpus = [
            "A man is eating food.",
            "A man is eating a piece of bread.",
            "The girl is carrying a baby.",
            "A man is riding a horse.",
            "A woman is playing violin.",
            "Two men pushed carts through the woods.",
            "A man is riding a white horse on an enclosed ground.",
            "A monkey is playing drums.",
            "A cheetah is running behind its prey.",
        ]
        model = CrossEncoderFactory('cross-encoder/ms-marco-MiniLM-L-6-v2')
        sentence_combinations = [[query, sentence] for sentence in corpus]
        scores = model.predict(sentence_combinations)
        print(scores)
        ranks = model.rank(query, corpus)

        # Print the scores
        print("Query:", query)
        for rank in ranks:
            print(f"{rank['score']:.2f}\t{corpus[rank['corpus_id']]}")


