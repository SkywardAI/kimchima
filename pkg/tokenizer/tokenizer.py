# This file is adapted from the LLama project:
# https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py

# Original LLama code by Facebook AI Research
# Adapted by Aisuko

from typing import List
import sentencepiece as spm

class Tokenizer:
    """
    A class for encoding and decoding text using SentencePiece tokenizer.

    Attributes:
    -----------
    sp_model : sentencepiece.SentencePieceProcessor
        The SentencePiece model used for encoding and decoding.
    n_words : int
        The size of the vocabulary.
    bos_id : int
        The ID of the beginning-of-sentence token.
    eos_id : int
        The ID of the end-of-sentence token.
    pad_id : int
        The ID of the padding token.
    """

    def __init__(self, model_path: str):
        """
        Initializes a Tokenizer object.

        Parameters:
        -----------
        model_path : str
            The path to the SentencePiece model file.
        """
        self.sp_model = spm.SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string using the SentencePiece model.

        Parameters:
        -----------
        s : str
            The string to be encoded.
        bos : bool
            Whether to add a beginning-of-sentence token to the beginning of the encoded sequence.
        eos : bool
            Whether to add an end-of-sentence token to the end of the encoded sequence.

        Returns:
        --------
        List[int]
            The encoded sequence as a list of integers.
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a sequence of integers using the SentencePiece model.

        Parameters:
        -----------
        t : List[int]
            The sequence of integers to be decoded.

        Returns:
        --------
        str
            The decoded string.
        """
        return self.sp_model.decode(t)