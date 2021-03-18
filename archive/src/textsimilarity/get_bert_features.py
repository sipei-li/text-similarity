import pandas as pd
import numpy as np
import os
import sys
from transformers import BertModel, BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


MAX_LENGTH=128
BATCH_SIZE=32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

vocab = tokenizer.get_vocab()
vocab_list = list(vocab.keys())


def pad_to_max_length(ids, MAX_LENGTH):
    ids = torch.tensor(pad_sequence(ids, batch_first=True))
    columns = ids.shape[1]
    output_ids = torch.zeros(ids.shape[0], MAX_LENGTH)
    output_ids[:, :columns] = ids
    return output_ids


def get_bert_embeddings(token_list):
    input_token_ids = list(map(lambda x: torch.tensor(tokenizer.convert_tokens_to_ids([tokenizer.tokenize(j)[0] for j in x])), token_list))
    input_token_ids = pad_to_max_length(input_token_ids, MAX_LENGTH)
    bert_embeddings = []
    dataset = TensorDataset(input_token_ids)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    with torch.no_grad():
        for b in dataloader:
            bert_embeddings.extend(bert_model(b[0].type(torch.LongTensor).to(device))[0].cpu().numpy())

    bert_embeddings = np.array(bert_embeddings)
    return bert_embeddings


if __name__ == '__main__':
    DATA_DIR = sys.argv[1]
    data_name = sys.argv[2]

    coref = np.load(os.path.join('data', DATA_DIR, 'interim', f'coref_{data_name}.npy'), allow_pickle=True)
    coref = list(map(lambda x: [j for n in x for j in n.split()], coref))

    bert_embeddings = get_bert_embeddings(coref)
    np.save(os.path.join('data', DATA_DIR, 'interim', f'bert_embeddings_{data_name}.npy'), bert_embeddings)




