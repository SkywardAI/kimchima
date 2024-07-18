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

from typing import Any
import torch
from torch.utils.tensorboard import SummaryWriter

class SimpleTrainer:
    def __init__(self, **kwargs):
        self.model: Any=kwargs.get('model') or None
        self.loss_func: Any=kwargs.get('loss_func') or None
        self.optimizer: Any=kwargs.get('optimizer') or None
        self.writer: Any = SummaryWriter()
        assert (self.model and self.loss_func and self.optimizer), "Model, Loss Function and Optimizer are required"

    def train(self, x:torch.Tensor, y:torch.Tensor)-> None:
        for epoch in range(10):
            y1 = self.model(x)
            loss = self.loss_func(y1, y)
            self.writer.add_scalar("Loss/train", loss, epoch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.writer.flush()
        self.writer.close()
