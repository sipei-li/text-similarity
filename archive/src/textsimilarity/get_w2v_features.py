import pandas as pd
import numpy as np
import os
import sys
import spacy
from textsimilarity.preprocess import get_coref_and_dp

nlp = spacy.load("en")

def get_word2vec_embeddings(doc, with_coref=True):
    if with_coref:
        coref = get_coref_and_dp.get_neural_coreference(doc)
        doc = ' '.join(coref)
    nlp_doc = nlp(doc)
    emb = np.array([nlp_doc[i].vector for i in range(len(doc.split()))])
    return emb

if __name__ == '__main__':
    DATA_DIR = sys.argv[1]
    data_name = sys.argv[2]

    data = pd.read_csv(os.path.join('data', DATA_DIR, f'raw/{data_name}.csv')).sentence
    # coref = np.load(os.path.join('data', DATA_DIR, 'interim', f'coref_{data_name}.npy'), allow_pickle=True)
    # coref = list(map(lambda x: [j for n in x for j in n.split()], coref))
    
    # MAX_LEN = max([len(line) for line in coref])
    # word2vec_embeddings = np.zeros((len(coref), MAX_LEN, 96), dtype=object)

    # for i in range(len(coref)):
    #     line = coref[i]
    #     length = len(line)
    #     word2vec_embeddings[i][:length] = np.array([nlp(w).vector for w in line])

    word2vec_embeddings = np.array(list(map(get_word2vec_embeddings, data)), dtype=object)

    np.save(os.path.join('data', DATA_DIR, 'interim', f'word2vec_embeddings_{data_name}.npy'), word2vec_embeddings)

