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

import os
import unittest
from pathlib import Path

from models.simple_gpt import SimpleGPT, SimpleGPTTrainer
from pkg.dataset_helper import DatasetHelper


class TestSimpleGPT(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        src_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
        abs_file_path = os.path.join(src_dir, "input.txt")
        _ = DatasetHelper.download_remote_file(SimpleGPTTrainer.ds_url, abs_file_path)

        cls.dataset = SimpleGPTTrainer.load_data(abs_file_path)
        cls.chars = SimpleGPTTrainer.unique_chars(cls.dataset)
        cls.vocabsize = SimpleGPTTrainer.build_vocab(cls.chars)
        cls.stoi = SimpleGPTTrainer.stoi(cls.chars)
        cls.itos = SimpleGPTTrainer.itos(cls.chars)

    def test_simple_gpt_trainer(self):
        encoder = SimpleGPTTrainer.encoder(self.stoi)
        # decoder=SimpleGPTTrainer.decoder(self.itos)

        ds = encoder(self.dataset)

        train_data, val_data = SimpleGPTTrainer.split_to_train_validate(ds, 0.9)

        model = SimpleGPT(self.vocabsize)
        optimizer = SimpleGPTTrainer.adam_optimizer(model)

        SimpleGPTTrainer.train(model, optimizer, train_data, val_data)

        output = SimpleGPTTrainer.sample(model, 100)

        decoder = SimpleGPTTrainer.decoder(self.itos)
        output = decoder(output)

        self.assertTrue(output)
