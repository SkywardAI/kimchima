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

from kimchima.pkg import PipelinesFactory
from kimchima.utils import Dialog

@unittest.skip("skip TestChatSummary test class")
class TestChatSummary(unittest.TestCase):

        conversation_model="gpt2"
        summarization_model="sshleifer/distilbart-cnn-12-6"
        msg = "why Melbourne is a good place to travel?"
        max_length = 10
        prompt = "Melbourne is often considered one of the most livable cities globally, offering a high quality of life."

        @classmethod
        def setUpClass(cls):
            # Load conversation model by using pipeline
            cls.pipe_con=PipelinesFactory.customized_pipe(model=cls.conversation_model, device_map='auto')
            cls.pipe_sum=PipelinesFactory.customized_pipe(model=cls.summarization_model, device_map='auto')

        
        def test_chat_summary(self):
            """
            Test chat_summary method
            """
            con = PipelinesFactory.init_conversation()
            con.add_message({"role": "user", "content": "Dod you like weather of Melbourne?"})
            con.add_message({"role": "assistant", "content": "Melbourne is also sunny which is my favourite weather"})
            con.add_message({"role": "user", "content": "why Melbourne is a good place to travel?"})
            res = Dialog.chat_summary(
                pipe_con=self.pipe_con,
                pipe_sum=self.pipe_sum,
                messages=con,
                prompt=self.prompt,
                max_length=self.max_length
                )

            # res is str and should not be None
            self.assertIsNotNone(res)
            self.assertIsInstance(res, str)
