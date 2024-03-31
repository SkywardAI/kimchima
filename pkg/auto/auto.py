# coding=utf-8
# Copyright [2024] [Aisuko]
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        
        Returns:
            None
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

        Returns:
            torch.tensor: The embeddings of text.
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
        """
        Mean pooling of model output.

        Args:
            model_output (torch.tensor): The model output.
            attention_mask (torch.tensor): The attention mask.

        Returns:
            torch.tensor: The mean pooling of model output.
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)