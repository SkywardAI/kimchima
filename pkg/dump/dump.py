import numpy as np
import torch
import pathlib


class Dump:
    """
    A class that provides methods to save various components of a transformer model.
    """
    @staticmethod
    def save_scalar(s, name, path):
        """
        Saves a scalar value to a numpy file.

        Args:
        - s: The scalar value to be saved.
        - name: The name of the file to be saved.
        - path: The path where the file will be saved.
        """
        s = np.array([1.0, float(s)]).astype(np.float32)
        np.save(pathlib.Path(path, f'{name}.npy'), s)
    
    @staticmethod
    def save_tensor(tensor, name, path):
        """
        Saves a tensor to a numpy file.

        Args:
        - tensor: The tensor to be saved.
        - name: The name of the file to be saved.
        - path: The path where the file will be saved.
        """
        tensor_numpy=tensor.numpy()
        tensor_dims =np.array(tensor_numpy.shape)
        tensor_values = tensor_numpy.flatten()
        tensor_to_save = np.concatenate((tensor_dims, tensor_values)).astype(np.float32)
        np.save(pathlib.Path(path, f'{name}.npy'), tensor_to_save)
    
    @staticmethod
    def save_linear(linear, path):
        """
        Saves the weight and bias of a linear layer to numpy files.

        Args:
        - linear: The linear layer to be saved.
        - path: The path where the files will be saved.
        """
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        Dump.save_tensor(linear.weight.t(), 'weight', path) # PyTorch and Tinygrad strangely transpose linear weights so reverse that
        if linear.bias is not None:
            Dump.save_tensor(linear.bias, 'bias', path)

    @staticmethod
    def save_rmsnorm(norm, path):
        """
        Saves the weight and epsilon value of a RMSNorm layer to numpy files.

        Args:
        - norm: The RMSNorm layer to be saved.
        - path: The path where the files will be saved.
        """
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        Dump.save_tensor(norm.weight, 'weight', path)
        Dump.save_scalar(norm.eps, 'eps', path)

    @staticmethod
    def save_attention(attention, path):
        """
        Saves the weight of the query, key, value and output linear layers of an attention layer to numpy files.

        Args:
        - attention: The attention layer to be saved.
        - path: The path where the files will be saved.
        """
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        Dump.save_linear(attention.wq, pathlib.Path(path, 'wq'))
        Dump.save_linear(attention.wk, pathlib.Path(path, 'wk'))
        Dump.save_linear(attention.wv, pathlib.Path(path, 'wv'))
        Dump.save_linear(attention.wo, pathlib.Path(path, 'wo'))
        n_kv_head = attention.n_kv_heads
        n_head = n_kv_head * attention.n_rep
        Dump.save_scalar(n_head, "n_head", path)
        Dump.save_scalar(n_kv_head, "n_kv_head", path)

    @staticmethod
    def save_feedforward(feed_forward, path):
        """
        Saves the weight of the three linear layers of a feedforward layer to numpy files.

        Args:
        - feed_forward: The feedforward layer to be saved.
        - path: The path where the files will be saved.
        """
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        Dump.save_linear(feed_forward.w1, pathlib.Path(path, 'w1'))
        Dump.save_linear(feed_forward.w2, pathlib.Path(path, 'w2'))
        Dump.save_linear(feed_forward.w3, pathlib.Path(path, 'w3'))

    @staticmethod
    def save_embedding(embedding, path):
        """
        Saves the weight of an embedding layer to a numpy file.

        Args:
        - embedding: The embedding layer to be saved.
        - path: The path where the file will be saved.
        """
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        Dump.save_tensor(embedding.weight, 'weight', path)

    @staticmethod
    def save_transformer_block(transformer_block, path):
        """
        Saves the components of a transformer block to numpy files.

        Args:
        - transformer_block: The transformer block to be saved.
        - path: The path where the files will be saved.
        """
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        Dump.save_attention(transformer_block.attention, pathlib.Path(path, 'attention'))
        Dump.save_feedforward(transformer_block.feed_forward, pathlib.Path(path, 'feedforward'))
        Dump.save_rmsnorm(transformer_block.attention_norm, pathlib.Path(path, 'attention_norm'))
        Dump.save_rmsnorm(transformer_block.ffn_norm, pathlib.Path(path, 'ffn_norm'))

    @staticmethod
    def save_transformer(transformer, path):
        """
        Saves the components of a transformer model to numpy files.

        Args:
        - transformer: The transformer model to be saved.
        - path: The path where the files will be saved.
        """
        with torch.no_grad():
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            Dump.save_scalar(len(transformer.layers), 'n_layer', path)
            for idx, layer in enumerate(transformer.layers):
                Dump.save_transformer_block(layer, pathlib.Path(path, f'layer{idx}'))
            Dump.save_rmsnorm(transformer.norm, pathlib.Path(path, 'norm'))
            Dump.save_embedding(transformer.tok_embeddings, pathlib.Path(path, 'tok_embeddings'))
            Dump.save_linear(transformer.output, pathlib.Path(path, 'output'))
            Dump.save_scalar(10000.0, 'theta', path)
            Dump.save_scalar(transformer.params.max_seq_len, 'n_ctx', path)
            Dump.save_scalar(transformer.params.multiple_of, 'multiple_of', path)
            if transformer.params.ffn_dim_multiplier is not None:
                Dump.save_scalar(transformer.params.ffn_dim_multiplier, 'ffn_dim_multiplier', path)
