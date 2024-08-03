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
import random
import unittest

from pathlib import Path
import torch
from torch.nn import functional as F

from models.mlp_batchnorm import MlpBatchNormTrainer,Linear, BatchNorm1d, Tanh
from pkg.dataset_helper import DatasetHelper


class TestMLPBatchNorm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_embd=MlpBatchNormTrainer.n_embed
        cls.n_hidden=MlpBatchNormTrainer.n_hidden
        cls.n_block_size=MlpBatchNormTrainer.n_block_size

        src_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
        abs_file_path = os.path.join(src_dir, "input.txt")
        _ = DatasetHelper.download_remote_file(MlpBatchNormTrainer.ds_url, abs_file_path)
        cls.data=MlpBatchNormTrainer.load_dataset(abs_file_path)
        cls.unique_chars=MlpBatchNormTrainer.unique_chars(cls.data.splitlines())
        cls.stoi=MlpBatchNormTrainer.stoi(cls.unique_chars)
        cls.itos=MlpBatchNormTrainer.itos(cls.unique_chars)
        cls.vocab_size=MlpBatchNormTrainer.build_vocab(cls.unique_chars)

    def test_mlp_batchnorm_trainer(self):
        random.seed(42)
        words=self.data.splitlines()
        random.shuffle(words)

        n1=int(0.8*len(words))
        n2=int(0.9*len(words))

        Xtr, Ytr=MlpBatchNormTrainer.build_dataset(words[:n1], self.stoi) # 80%
        Xdev, Ydev=MlpBatchNormTrainer.build_dataset(words[n1:n2],self.stoi) # 10%
        Xte, Yte=MlpBatchNormTrainer.build_dataset(words[n2:],self.stoi) # 10%
        g=torch.Generator().manual_seed(2147483647)

        C=torch.randn((self.vocab_size, self.n_embd), generator=g)

        # sequential 6 MLP layers
        layers=[
            Linear(self.n_embd*self.n_block_size, self.n_hidden, bias=False), BatchNorm1d(self.n_hidden), Tanh(),
            Linear(self.n_hidden, self.n_hidden, bias=False), BatchNorm1d(self.n_hidden), Tanh(),
            Linear(self.n_hidden, self.n_hidden, bias=False), BatchNorm1d(self.n_hidden), Tanh(),
            Linear(self.n_hidden, self.n_hidden, bias=False), BatchNorm1d(self.n_hidden), Tanh(),
            Linear(self.n_hidden, self.n_hidden, bias=False), BatchNorm1d(self.n_hidden), Tanh(),
            Linear(self.n_hidden, self.vocab_size, bias=False), BatchNorm1d(self.vocab_size)
        ]

        with torch.no_grad():
            # here, out latest layer is a batch norm layer and we wouldn't change the weights to make the softmax less confident
            # we would like to changing the gamma(from the batch norm paper algorithm1)
            # because gamma remember int he batchnorm is the variable that multiplicatively interacts with the output of thah normalization
            layers[-1].gamma*=0.1

            # all pther layers: apply again
            for layer in layers[:-1]:
                if isinstance(layer, Linear):
                    layer.weight*=5/3 # booster the linear layer by the gain, the number from torch document
        # [C] the embedding matrix and all the parameters of all the layers
        parameters=[C]+[p for layer in layers for p in layer.parameters()]
        print(sum(p.nelement() for p in parameters)) # number of parameters in total
        for p in parameters:
            p.requires_grad=True

        
        # training loop
        lossi=[]
        ud=[]

        for i in range(MlpBatchNormTrainer.max_steps):
            # minibatch construct
            ix=torch.randint(0, Xtr.shape[0], (MlpBatchNormTrainer.batch_size,), generator=g)
            Xb, Yb=Xtr[ix], Ytr[ix] # batch X,Y

            # forward pass
            emb= C[Xb] # embed the characters into vectors
            x=emb.view(emb.shape[0], -1) # flatten/concatenate the vectors
            for layer in layers:
                x=layer(x)
            loss=F.cross_entropy(x, Yb) # loss function

            # backward pass
            for layer in layers:
                layer.out.retain_grad()
            
            for p in parameters:
                p.grad=None
            
            loss.backward()

            # update
            lr=0.1 if i<100000 else 0.01 # step learning rate decay
            for p in parameters:
                p.data+=-lr*p.grad
            
            # track stats
            if i%10000==0: # print every once in a while
                print(f'{i:7d}/{MlpBatchNormTrainer.max_steps:7d}: {loss.item():.4f}')
            lossi.append(loss.log10().item())
            
            with torch.no_grad():
                ud.append([(lr*p.grad.std()/p.data.std()).log10().item() for p in parameters])
            
            if i>=1000:
                break


        g=torch.Generator().manual_seed(2147483647+10)

        for _ in range(20):
            out=[]
            context=[0]*self.n_block_size
            while True:
                #forward pass the neural net
                emb=C[torch.tensor([context])] # (1, block_size, n_embd)
                x=emb.view(emb.shape[0], -1) # concatenate the vectors
                for layer in layers:
                    x=layer(x)
                logits=x
                probs=F.softmax(logits, dim=-1)
                # sample from the distribution
                ix=torch.multinomial(probs, num_samples=1, generator=g).item()
                # shift the contetx window and track the samples
                context=context[1:]+[ix]
                out.append(ix)
                if ix==0:
                    break
            print(''.join(self.itos[i] for i in out[:-1]))