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
#https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py


from torch import nn
from torch import Tensor
import torch
import logging
from tqdm.autonotebook import trange
from torch.nn import CrossEntropyLoss
import numpy as np

from models.Pooling import Pooling
from models.Transformer import Transformer


class SiameseNet(nn.Module):
    """
        At training, a SiameseNet takes exactly two sentences / texts as input and either predicts
        a score or label for this sentence pair. It can for example predict the probability of the 
        second sentence being the next sentence of the first sentence on a scale of 0 ... 1.
        At inteference time, a SiameseNet takes one sentence to encode it into a vector.
        """
    def __init__(self, args, model):
        super().__init__()
        self.model0 = model.to(device=args.device)
        self.config = self.model0.config
        if args.train_both_branches == True:
            self.model1 = Transformer(args)
            for param in self.model1.parameters():
                param.requires_grad = False
        self.embedding_size = self.config.hidden_size
        self.pooling = Pooling(args, self.embedding_size).to(device=args.device)
        self.linear = nn.Linear(self.embedding_size*2, 2)
        self.args = args
    
    def encode(self, sentences,
            show_progress_bar: bool = True,
            output_value: str = 'sentence_embedding',
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            is_pretokenized: bool = False):
        
        """
        Computes sentence embeddings
        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param is_pretokenized: DEPRECATED - No longer used, will be removed in the future
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        
        if convert_to_tensor:
            convert_to_numpy = False

        self.to(self.args.device)

        all_embeddings = []
        for start_index in trange(0, len(sentences), self.args.batch_size, desc="Batches"):
            sentences_batch = sentences[start_index:start_index+self.args.batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, self.args.device)

            with torch.no_grad():
                out_features = self.model0(features)
                out_features = self.pooling(out_features)
                embeddings = out_features[output_value]

                if output_value == 'token_embeddings':
                    #Set token embeddings to 0 for padding tokens
                    input_mask = out_features['attention_mask']
                    input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings = embeddings * input_mask_expanded

                embeddings = embeddings.detach()

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        return all_embeddings

    def forward(self, features1, features2, labels = None):
        # TODO： consider making the second model non-trainable
        if not self.args.train_both_branches:
            outputs1 = self.model0(features1)
            outputs2 = self.model0(features2)
        else:
            if self.args.phrase_from_first_sent:
                outputs1 = self.model0(features1)
                outputs2 = self.model1(features2)
            else:
                outputs1 = self.model1(features1)
                outputs2 = self.model0(features2)
                
        outputs1 = self.pooling(outputs1)
        outputs2 = self.pooling(outputs2)
        concat_vector = torch.cat((outputs1['sentence_embedding'], outputs2['sentence_embedding']), -1)
        logits = self.linear(concat_vector)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return next_sentence_loss, logits if next_sentence_loss is not None else logits

    def get_max_seq_length(self):
        """
        Returns the maximal sequence length for input the model accepts. Longer inputs will be truncated
        """
        if hasattr(self._first_module(), 'max_seq_length'):
            return self._first_module().max_seq_length

        return None

    def tokenize(self, text: str):
        """
        Tokenizes the text
        """
        return self._first_module().tokenize(text)

    def get_sentence_features(self, *features):
        return self._first_module().get_sentence_features(*features)

    def get_sentence_embedding_dimension(self):
        for mod in reversed(self._modules.values()):
            sent_embedding_dim_method = getattr(mod, "get_sentence_embedding_dimension", None)
            if callable(sent_embedding_dim_method):
                return sent_embedding_dim_method()
        return None

    def _first_module(self):
        """Returns the first module of this sequential embedder"""
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        """Returns the last module of this sequential embedder"""
        return self._modules[next(reversed(self._modules))]

        
def batch_to_device(batch, device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(device)
    return batch