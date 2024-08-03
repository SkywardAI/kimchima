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


import random
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# https://www.kaggle.com/code/aisuko/implement-neural-net-with-batch-norm-layer

g=torch.Generator().manual_seed(2147483647)

class Linear:
    """
    Linear layer
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight=torch.randn((fan_in, fan_out), generator=g)/fan_in**0.5 # unit gaussian
        self.bias=torch.zeros(fan_out) if bias else None # default bias initialize to zeros

    def __call__(self, x):
        self.out=x@self.weight
        if self.bias is not None:
            self.out+=self.bias
        return self.out
    
    def parameters(self):
        """
        return tensors that are parameters of this layer
        """
        return [self.weight]+([] if self.bias is None else [self.bias])


class BatchNorm1d:
    """
    batchnorm layer
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html formula
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps=eps # used in division
        self.momentum=momentum # keep tracking running stats
        self.training=True

        # parameters (trained with backprop)
        self.gamma=torch.ones(dim)
        self.beta=torch.zeros(dim)
        # buffers (trained with a running 'momentum update')
        self.running_mean=torch.zeros(dim)
        self.running_var=torch.ones(dim)
    
    def __call__(self, x):
        """
        Follow https://arxiv.org/pdf/1502.03167

        Algorithm 1
        1.mini-batch mean
        2.mini-batch variance
        3.normalize
        4.scale and shift
        """

        # calculating the forward pass
        if self.training:
            xmean=x.mean(0, keepdim=True) # batch mean
            xvar=x.var(0, keepdim=True, unbiased=True) # batch variance
        else:
            xmean=self.running_mean
            xvar=self.running_var
        
        xhat=(x-xmean)/torch.sqrt(xvar+self.eps) # normalize to unit variance
        self.out=self.gamma*xhat+self.beta # craete otu attribute for visualization training process
        # update the buffers

        if self.training:
            with torch.no_grad():
                self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*xmean
                self.running_var=(1-self.momentum)*self.running_var+self.momentum*xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:
    """
    """
    def __call__(self, x):
        self.out=torch.tanh(x)
        return self.out

    def parameters(self):
        """
        no parameters in this layer
        """
        return []


class MlpBatchNormTrainer:
    """
    MlpBatchNormTrainer
    """
    ds_url="https://www.kaggleusercontent.com/kf/187064505/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..ZLA9V0sWqB_Px0312U15fQ.OO2wvSdp-fhBB0BTDAaLToek6CLGlzS4otHIsyHBd1feEJxIUq055-GIQb24Ez51pGq31hyzaN_vFeDRnqxFwyc12sDNqZ7uDhel-5xeXU08h0qNtOpoqXA-iJpPuV4u-dThq8Lk-zoOg_ZDmVNAW8XHZVAM2ZAHl9StyqN1n7eOGU0379mp_2ol2gyjXP01xNDH2n4kUSIIetktnagIon8Jm_tcLBB-DaWPTFwQ5L7NBP1t-omCUrKydTxAyPIFFwnid3T1vzEgSmYiUY8Ec-iC8OG5d2pKcod9FIAOkJH4Xu74Pvzp5UuOFzQXRByezEOkyD0ltAhfMOab0ebIi6YSTVKrna70HZhuxjWQRK9fIgvt0V7RMz84ZQspJWrgofowQrf7E1avVvXe7GQW4E7dYITqQoJvZ7dhlpujq1db6pkegRqOfuQzPJcD6UHBTpVRyi36rIQoLpFd63XLzY5eya4ScAy5H-frQhF0IU927Z86S9iR2AypqO3TXriPsMHjJ7o-DwXpnHCNkVfMJXeVxT36DRBiV9uCTL-e8_xOUKw50N5iG3NqTnos0IwSXvwrSBtHxUI71zo-I2Z-l5x_GqjEa9QVl1XX_q7GU_YFejlC-rT9KdcA_6TEVO6qaMpfvVvCc9kFYI7s7GQNbg.tIuWJu1a71qSZKZeG-TgPg/names.txt"
    n_embed=10 # the dimensionality of the character embedding vectors
    n_hidden=100 # the number of neurons in the hidden layer of the MLP
    n_block_size=3 # context length: how many characters do we take to predict the next one?
    max_steps=200000
    batch_size=32


    def __init__(self):
        raise Exception("This class is not meant to be instantiated")
    
    @classmethod
    def set_hyperparameters(cls, **kwargs):
        """
        Set hyperparameters
        """
        cls.n_embed=kwargs.get("n_embed", 10)
        cls.n_hidden=kwargs.get("n_hidden", 100)
        cls.n_block_size=kwargs.get("n_block_size", 3)
        cls.max_steps=kwargs.get("max_steps", 200000)
        cls.batch_size=kwargs.get("batch_size", 32)
    
    @classmethod
    def load_dataset(cls, filePath:str)->str:
        """
        Load the dataset
        """
        with open(filePath, "r", encoding="utf-8") as f:
            text=f.read()
        return text
    
    @classmethod
    def unique_chars(cls, text:str)->list:
        """
        Get all the unique characters in the text
        """
        return sorted(list(set(''.join(text))))
    
    @classmethod
    def stoi(cls, chars:list)->dict:
        """
        Convert characters to indices
        """
        stoi={char:i+1 for i,char in enumerate(chars)}
        stoi['.']=0
        return stoi

    @classmethod
    def itos(cls, chars:list)->dict:
        """
        Convert indices to characters
        """
        itos={i:char for char,i in cls.stoi(chars).items()}
        return itos
    
    @classmethod
    def build_vocab(cls, chars:list)->int:
        """
        Build a vocabulary from the unique characters
        """
        return len(chars)
    
    @classmethod
    def build_dataset(cls, words:str, stoi: dict)->tuple[torch.Tensor, torch.Tensor]:
        """
        Build the dataset
        """
        X,Y=[],[]

        for w in words:
            context=[0]*cls.n_block_size
            for ch in w+'.':
                ix=stoi[ch]
                X.append(context)
                Y.append(ix)
                context=context[1:]+[ix] # crop and append

        X=torch.tensor(X) # convert to tensor
        Y=torch.tensor(Y)
        return X,Y
