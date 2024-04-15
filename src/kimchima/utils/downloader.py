
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

from kimchima.pkg import logging
from transformers import pipeline

logger=logging.get_logger(__name__)



class Downloader:

    def __init__(self):
        raise EnvironmentError(
            "Embeddings is designed to be instantiated "
            "using the `Embeddings.from_pretrained(pretrained_model_name_or_path)` method."
        )
    
    @classmethod
    def model_downloader(cls, *args, **kwargs)->str:
        r"""
        """
        model_name=kwargs.pop("model_name", None)
        if model_name is None:
            raise ValueError("model_name is required")

        folder_name=kwargs.pop("folder_name", None)
        pipe=pipeline(model=model_name)
        pipe.save_pretrained(folder_name if folder_name is not None else model_name)
