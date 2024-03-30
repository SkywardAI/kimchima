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

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class Auto:
    """
    A class for load model and it's tokenizer by using model name or path of model.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the class with model name or path of model.
        
        Args:
            model_name_or_path (str): The model name or path of model.
        """
        model_name_or_path = kwargs.pop('model_name_or_path', None)
        if model_name_or_path is None:
            return None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # https://github.com/huggingface/transformers/blob/v4.39.2/src/transformers/models/auto/auto_factory.py#L444
        # TODO @aisuko: add more parameters
        self.model = AutoModel.from_pretrained(model_name_or_path)

    def get_embeddings(self, *args, **kwargs):
        """
        Get embeddings of text.

        Args:
            text (str): The text to get embeddings.
            device (str): The device to use. Default is 'cpu'.
            max_length (int): The maximum length of text. Default is 512.
        """
        try:
            text = kwargs.pop('text', None)
            device = kwargs.pop('device', 'cpu')
            max_length = kwargs.pop('max_length', 512)

            inputs_ids = self.tokenizer(text, return_tensors='pt',max_length=max_length, padding=True, truncation=True).to(device)

            with torch.no_grad():
                output = self.model(**inputs_ids)
            
            embeddings=Auto.mean_pooling(output, inputs_ids['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(embeddings, p=2, dim=1)

        except Exception as e:
            sentence_embeddings=None
        return sentence_embeddings

    @staticmethod
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)