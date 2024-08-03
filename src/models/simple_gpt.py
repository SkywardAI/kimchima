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


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

# https://www.kaggle.com/code/aisuko/gpt-from-scratch-as-a-script

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both(B,T) tensors of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)  # B*T it also ok here
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)  # call forward automatically
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx


class SimpleGPTTrainer:
    ds_url = "https://www.kaggleusercontent.com/kf/189948176/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..1Jb9UiqqB4H5KVymvfAOrw.cX1aqGGJcBvzxM56ysKgmNbqhQdDIr5UuLBnx2OOHQJlOAAZMwG4n27TKm2K-KN7cxSiUxsLV-Ua53hQa7Y-Eup4QhqYRs47y_IFRVHxqUYILGfbzcZHaTtdvZM2UlGcMjO3-htDg3huWl_bT6vD0wEIpWWjw_vFA8MBiFndQUgBQcjnwMI4W-KKfOpeKcaonl-3HLaIBoDau-fGAFq1KPY7h6M1Oy20c4goF86AGyVYC1E3rbipDcIuF5jLjiUXLh6B5TgpybwmygfdsKrz8qOoK0W2UFEwH0pNQ1a3le222k1s7iwnLofU7P0cznFKa4glCa6U7UQ4JMcB371Pcz9YQXA5f8dvfOymgpFQ7Jwjx6FJZ211bD3zHYq2RYM1pE5N_0U-iPOnAHlNKVSgnOWbGkaJtckDUa7MHgfbJEEcPMjPdEZRf1AofQJKoFK3QTH87wpjboUxo8F-KfKr-40K5HbNisTOuJbSeZrBE1y1EDvbBQ1rFQxei9bjyz71eZdV9pjwdYEso1C1M8I669mAGmJ4X9TDkl2eO3wItIZzE5Jy5CIug8j6-kghz-jBDr9wkiMiwWoZ3rcM8JM1dbPDV-8HDTBfiAZFDl5w4tLH8o7bKXbd004X3l4H-O5uIj0inEv07OsU-80CSzkuuQ._myoCNJ7mrE61Hp6wJtDbw/input.txt"

    # hyperaparameters
    batch_size = 32  # how many independent sequences will we process in parallel
    block_size = 8  # what is the maximum context length got predictions?
    max_iters = 100
    eval_interval = 300
    learning_rate = 1e-2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_iters = 200

    def __init__(self) -> None:
        raise Exception("This class is not meant to be instantiated")

    @classmethod
    def set_hyperparameters(cls, **kwargs) -> None:
        """
        Set hyperparameters
        """
        cls.batch_size = kwargs.get("batch_size", 32)
        cls.block_size = kwargs.get("block_size", 8)
        cls.max_iters = kwargs.get("max_iters", 3000)
        cls.eval_interval = kwargs.get("eval_interval", 300)
        cls.learning_rate = kwargs.get("learning_rate", 1e-2)
        cls.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        cls.eval_iters = kwargs.get("eval_iters", 200)

    @classmethod
    def load_data(cls, ds_file) -> str:
        """
        Load the dataset
        """
        with open(ds_file, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    @classmethod
    def unique_chars(cls, text: str) -> list:
        """
        Get all the unique characters in the text
        """
        return sorted(list(set(text)))

    @classmethod
    def build_vocab(cls, chars: list) -> int:
        """
        Build a vocabulary from the unique characters
        """
        return len(chars)

    @classmethod
    def stoi(cls, chars: list) -> dict:
        """
        Convert characters to indices
        """
        return {char: i for i, char in enumerate(chars)}

    @classmethod
    def itos(cls, chars: list) -> dict:
        """
        Convert indices to characters
        """
        return {i: char for i, char in enumerate(chars)}

    @classmethod
    def encoder(cls, stoi: dict) -> torch.Tensor:
        """
        Convert string to list of indices
        """
        return lambda s: [stoi[c] for c in s]

    @classmethod
    def decoder(cls, itos: dict) -> torch.Tensor:
        """
        Convert list of indices to string
        """
        return lambda x: "".join([itos[i] for i in x])

    @classmethod
    def split_to_train_validate(
        cls, text: torch.Tensor, train_frac: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split the text into training and validation sets
        """
        data = torch.tensor(
            text, dtype=torch.long
        )  # construct a tensor with no autograd history
        n = int(train_frac * len(data))
        train_data = data[:n]
        val_data = data[n:]
        return train_data, val_data

    @classmethod
    def get_batch(
        cls, split: str, train_data: torch.Tensor, val_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a small batch of data of inputs x and targets y
        """
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - cls.block_size, (cls.batch_size,))
        x = torch.stack([data[i : i + cls.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + cls.block_size + 1] for i in ix])
        x, y = x.to(cls.device), y.to(cls.device)
        return x, y

    @classmethod
    def adam_optimizer(cls, model: SimpleGPT) -> torch.optim.Adam:
        """
        Create an optimizer
        """
        return torch.optim.Adam(model.parameters(), lr=cls.learning_rate)

    @classmethod
    def train(
        cls,
        model: SimpleGPT,
        optimizer: torch.optim.Adam,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
    ) -> float:
        """
        Train the model
        """

        @torch.no_grad()
        def estimate_loss():
            out = {}
            model.eval()
            for split in ["train", "val"]:
                losses = torch.zeros(cls.eval_iters)
                for k in range(cls.eval_iters):
                    X, Y = cls.get_batch(split, train_data, val_data)
                    logits, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            model.train()
            return out

        writer = SummaryWriter()
        torch.manual_seed(1337)

        for i in range(cls.max_iters):
            # every once in a while evaluate the loss on train and val sets
            if i % cls.eval_interval == 0:
                losses = estimate_loss()
                print(
                    f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                writer.add_scalar("Loss/train", losses["train"], i)

            # sample a batch of data
            xb, yb = cls.get_batch("train", train_data, val_data)

            # evalute the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        writer.flush()
        writer.close()

    @classmethod
    def sample(cls, model: SimpleGPT, max_new_tokens: int) -> list:
        """
        Getting the sample from the model
        """
        context = torch.zeros((1, 1), dtype=torch.long, device=cls.device)
        return model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
