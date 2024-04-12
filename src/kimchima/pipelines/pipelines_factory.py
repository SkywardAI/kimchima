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


class PipelinesFactory:
    r"""
    A factory class for creating Huggingface Transformers pipelines for different ML tasks.
    """

    def __init__(self):
        raise EnvironmentError(
            "Pipelines is designed to be instantiated "
            "using the `Pipelines.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def text_generation(cls, *args,**kwargs)-> pipeline:
        r"""
        Create a text generation pipeline using the Huggingface Transformers library.
        """
        
        model=kwargs.pop("model", None)
        if model is None:
            raise ValueError("model is required")
        tokenizer=kwargs.pop("tokenizer", None)
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        streamer=kwargs.pop("text_streamer", None)
        max_new_tokens=kwargs.pop("max_new_tokens", 20)
        
        #
        pipe=pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            device_map='auto',
            **kwargs
        )

        logger.debug(f"The text generation pipeline device is {pipe.device}")

        return pipe
    
    @classmethod
    def customized_pipe(cls, *args,**kwargs)-> pipeline:
        r"""
        Create a customized pipeline using the Huggingface Transformers library.
        With any param which is allowd in the pipeline function.
        """
        
        #
        pipe=pipeline(
            **kwargs
        )
        return pipe
    
    @classmethod
    def chat_response(cls, *args,**kwargs)-> str:
        r"""
        Get the chat response based on converation messages and suggestion answer
        """
        conversation_model=kwargs.pop("conversation_model", None)
        if conversation_model is None:
            raise ValueError("conversation_model is required")
        summarization_model=kwargs.pop("summarization_model", "summarization")
        messages=kwargs.pop("messages", None)
        if messages is None:
            raise ValueError("messages is required")
        prompt=kwargs.pop("prompt", None)
        max_length=kwargs.pop("max_length", None)
        
        #
        chatbot = pipeline(model=conversation_model)
        response = chatbot(messages)
        
        if prompt is None:
            return response[0].get('generated_text')
        
        raw_response = prompt + response[0].get('generated_text')
        
        if max_length is None:
            max_length = len(raw_response)

        summarizer = pipeline(summarization_model)
        response = summarizer(raw_response, min_length=5, max_length=max_length)
        
        return response[0].get('summary_text')