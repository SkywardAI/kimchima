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

import unittest

from kimchima.pkg import Auto


class TestAuto(unittest.TestCase):

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    def test_get_embeddings(self):
        """
        Test get_embeddings method
        """
        model = Auto(model_name_or_path=self.model_name)
        embeddings = model.get_embeddings(text='Melbourne')
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape, (1, 384))


    def test_get_embeddings_with_list(self):
        """
        Test get_embeddings method with list of text
        """
        model = Auto(model_name_or_path=self.model_name)
        embeddings = model.get_embeddings(text=['Melbourne', 'Sydney'])
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape, (2, 384))

