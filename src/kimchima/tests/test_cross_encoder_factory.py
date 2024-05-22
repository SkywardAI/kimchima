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

from kimchima.pkg import (
    CrossEncoderFactory
)

@unittest.skip("skip CrossEncoderFactory")
class TestCrossEncoderFactory(unittest.TestCase):

    encoder_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    query = "A man is eating pasta."

    # With all sentences in the corpus
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ]
    
    @classmethod
    def setUpClass(cls):
        cls.encoder = CrossEncoderFactory(cls.encoder_name)

    def test_predict(self):
        """
        Test predict method
        """

        self.assertIsNotNone(self.encoder)
        sentence_combinations = [[self.query, sentence] for sentence in self.corpus]
        scores = self.encoder.predict(sentence_combinations)
        self.assertIsNotNone(scores)
        
    def test_rank(self):
        """
        Test rank method
        """

        ranks = self.encoder.rank(self.query, self.corpus)

        self.assertIsNotNone(ranks)



