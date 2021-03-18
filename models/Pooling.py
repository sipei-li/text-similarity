#####################################################################
############ Code sourced from Sentence-BERT repository##############
# @inproceedings{reimers-2019-sentence-bert,
#     title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
#     author = "Reimers, Nils and Gurevych, Iryna",
#     booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
#     month = "11",
#     year = "2019",
#     publisher = "Association for Computational Linguistics",
#     url = "https://arxiv.org/abs/1908.10084",
# }
#https://github.com/UKPLab/sentence-transformers/blob/09c557b8b3a0618f8502550efa0c88af7fc92432/sentence_transformers/models/Pooling.py



import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json


class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.
    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.
    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but devide by sqrt(input_length).
    """
    def __init__(self, args,
                 word_embedding_dimension: int
                 ):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension',  'pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

        self.word_embedding_dimension = word_embedding_dimension
        
        self.pooling_mode_cls_token = False
        self.pooling_mode_mean_tokens = False
        self.pooling_mode_max_tokens = False
        self.pooling_mode_mean_sqrt_len_tokens = False

        if args.pooling == 'mean':
            self.pooling_mode_mean_tokens = True
        elif args.pooling == 'max':
            self.pooling_mode_max_tokens = True
        elif args.pooling == 'cls':
            self.pooling_mode_cls_token = True
        elif args.pooling == 'meansqrt':
            self.pooling_mode_mean_sqrt_len_tokens = True

        self.pooling_output_dimension = word_embedding_dimension

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        if self.pooling_mode_cls_token:
            output_vector = cls_token
        elif self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value

            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vector = max_over_time
        elif self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())                
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vector = sum_embeddings / sum_mask
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vector = sum_embeddings / torch.sqrt(sum_mask)

        # output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension
