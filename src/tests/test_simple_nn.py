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
import torch
from models.simple_nn import SimpleNN
from trainers.simple_trainer import SimpleTrainer


class TestSimpleNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = torch.nn.Linear(1, 1)
        cls.criterion = torch.nn.MSELoss()
        cls.optimizer = torch.optim.SGD(cls.model.parameters(), lr=0.1)

    
    def test_simple_trainer(self):
        x = torch.arange(-5, 5, 0.1).view(-1, 1)
        y = -5 * x + 0.1 * torch.randn(x.size())
        trainer=SimpleTrainer(model=self.model, loss_func=self.criterion, optimizer=self.optimizer)
        self.assertTrue(trainer)
        trainer.train(x, y)

