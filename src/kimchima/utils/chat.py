
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
from kimchima.pipelines import PipelinesFactory


logger=logging.get_logger(__name__)



def chat_summary(*args,**kwargs)-> str:
        r"""
        Create a chat response and summarization pipeline using the Huggingface Transformers library.
        """
        conversation_model=kwargs.pop("conversation_model", None)
        if conversation_model is None:
            raise ValueError("conversation_model is required")
        summarization_model=kwargs.pop("summarization_model", None)
        messages=kwargs.pop("messages", None)
        if messages is None:
            raise ValueError("messages is required")
        prompt=kwargs.pop("prompt", None)
        max_length=kwargs.pop("max_length", None)
        
        # text generation pipeline
        pipe=PipelinesFactory.customized_pipe(
             model=conversation_model, 
             device_map='auto'
        )
        response = pipe(messages)
        
        if prompt is None:
            return response[0].get('generated_text')
        
        raw_response = prompt + response[0].get('generated_text')
        
        if max_length is None:
            max_length = len(raw_response)


        # pipeline for summarization
        pipe=PipelinesFactory.customized_pipe(
             model=summarization_model, 
             device_map='auto'
        )

        response = pipe(raw_response, min_length=5, max_length=max_length)

        return response[0].get('summary_text')
