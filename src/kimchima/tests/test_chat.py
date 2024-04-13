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

import unittest

from kimchima.utils.chat import chat_summary

class TestChatSummary(unittest.TestCase):
        
        @unittest.skip("skip test_chat_summary")
        def test_chat_summary(self):
            """
            Test chat_summary method
            """
            conversation_model="gpt2"
            summarization_model="sshleifer/distilbart-cnn-12-6"
            msg = "why Melbourne is a good place to travel?"
            prompt = "Melbourne is often considered one of the most livable cities globally, offering a high quality of life."

            res = chat_summary(
                conversation_model=conversation_model,
                summarization_model=summarization_model,
                messages=msg,
                prompt=prompt
                )

            # res is str and should not be None
            self.assertIsNotNone(res)
            self.assertIsInstance(res, str)