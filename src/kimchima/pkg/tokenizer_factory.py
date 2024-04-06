# coding=utf-8
# Copyright [2024] [SkywardAI]
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from transformers import AutoTokenizer
from kimchima.pkg import logging

logger = logging.get_logger(__name__)


class TokenizerFactory:
    r"""

    TokenizerFactory class to get the tokenizer from the specified model.
    
    Args:
        pretrained_model_name_or_path: pretrained model name or path

    """
    def __init__(self):
        raise EnvironmentError(
            "TokenizerFactory is designed to be instantiated "
            "using the `TokenizerFactory.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    def auto_tokenizer(cls, pretrained_model_name_or_path, **kwargs)-> AutoTokenizer:
        r"""
        """
        if pretrained_model_name_or_path is None:
            raise ValueError("pretrained_model_name_or_path cannot be None")

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        logger.debug(f"Loaded tokenizer: {pretrained_model_name_or_path}")
        return tokenizer