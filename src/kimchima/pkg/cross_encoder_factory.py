from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from kimchima.pkg import logging
import platform
from typing import Dict, List, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers.utils import PushToHubMixin


logger=logging.get_logger(__name__)

"""
Original code from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/cross_encoder/CrossEncoder.py
Adapted by: Rob Zhang
Date: 20240523
"""

class CrossEncoderFactory(PushToHubMixin):
    """
    A CrossEncoder takes exactly two sentences / texts as input and either predicts
    a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
    on a scale of 0 ... 1.

    It does not yield a sentence embedding and does not work for individual sentences.

    :param model_name: A model name from Hugging Face Hub that can be loaded with AutoModel, or a path to a local
        model. We provide several pre-trained CrossEncoder models that can be used for common tasks.
    :param num_labels: Number of labels of the classifier. If 1, the CrossEncoder is a regression model that
        outputs a continuous score 0...1. If > 1, it output several scores that can be soft-maxed to get
        probability scores for the different classes.
    :param max_length: Max length for input sequences. Longer sequences will be truncated. If None, max
        length of the model will be used
    :param device: Device that should be used for the model. If None, it will use CUDA if available.
    :param tokenizer_args: Arguments passed to AutoTokenizer
    :param automodel_args: Arguments passed to AutoModelForSequenceClassification
    :param trust_remote_code: Whether or not to allow for custom models defined on the Hub in their own modeling files.
        This option should only be set to True for repositories you trust and in which you have read the code, as it
        will execute code present on the Hub on your local machine.
    :param revision: The specific model version to use. It can be a branch name, a tag name, or a commit id,
        for a stored model on Hugging Face.
    :param local_files_only: If `True`, avoid downloading the model.
    :param default_activation_function: Callable (like nn.Sigmoid) about the default activation function that
        should be used on-top of model.predict(). If None. nn.Sigmoid() will be used if num_labels=1,
        else nn.Identity()
    :param classifier_dropout: The dropout ratio for the classification head.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = None,
        max_length: int = None,
        device: str = None,
        tokenizer_args: Dict = {},
        automodel_args: Dict = {},
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        classifier_dropout: float = None,
    ):
        self.config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, revision=revision, local_files_only=local_files_only
        )
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any(
                [arch.endswith("ForSequenceClassification") for arch in self.config.architectures]
            )

        if classifier_dropout is not None:
            self.config.classifier_dropout = classifier_dropout

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            **automodel_args,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **tokenizer_args,
        )
        self.max_length = max_length

        if device is None:
            if platform.system() == 'Darwin':
                device='mps'
            elif torch.cuda.is_available():
                device='cuda'
            else:
                device='cpu'
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    def _smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(
            *texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.max_length
        )

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized

    def predict(
        self,
        sentences: List[List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        num_workers: int = 0,
        activation_fct=None,
        apply_softmax=False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.

        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor: Convert the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(
            sentences,
            batch_size=batch_size,
            collate_fn=self._smart_batching_collate_text_only,
            num_workers=num_workers,
            shuffle=False,
        )

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores

    def rank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = False,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        num_workers: int = 0,
        activation_fct=None,
        apply_softmax=False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> List[Dict]:
        """
        Performs ranking with the CrossEncoder on the given query and documents. Returns a sorted list with the document indices and scores.

        Example:
            ::

                from sentence_transformers import CrossEncoder
                model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

                query = "Who wrote 'To Kill a Mockingbird'?"
                documents = [
                    "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
                    "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
                    "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961.",
                    "Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.",
                    "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era.",
                    "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."
                ]

                model.rank(query, documents, return_documents=True)

            ::

                [{'corpus_id': 0,
                'score': 10.67858,
                'text': "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature."},
                {'corpus_id': 2,
                'score': 9.761677,
                'text': "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961."},
                {'corpus_id': 1,
                'score': -3.3099542,
                'text': "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil."},
                {'corpus_id': 5,
                'score': -4.8989105,
                'text': "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."},
                {'corpus_id': 4,
                'score': -5.082967,
                'text': "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era."}]

        :param query: A single query
        :param documents: A list of documents
        :param top_k: Return the top-k documents. If None, all documents are returned.
        :param return_documents: If True, also returns the documents. If False, only returns the indices and scores.
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor: Convert the output to a tensor.
        :return: A sorted list with the document indices and scores, and optionally also documents.
        """
        query_doc_pairs = [[query, doc] for doc in documents]
        scores = self.predict(
            query_doc_pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            num_workers=num_workers,
            activation_fct=activation_fct,
            apply_softmax=apply_softmax,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
        )

        results = []
        for i in range(len(scores)):
            if return_documents:
                results.append({"corpus_id": i, "score": scores[i], "text": documents[i]})
            else:
                results.append({"corpus_id": i, "score": scores[i]})

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]
