import numpy as np
import pandas as pd
import os
# import nltk
import sys
import codecs
import json
from tqdm import tqdm
#from allennlp.predictors.predictor import Predictor
#import allennlp_models.rc
from nltk import word_tokenize
import spacy
import neuralcoref
nlp = spacy.load("en")
neuralcoref.add_to_pipe(nlp)

pronouns = ['it', 'its', 'he', 'him', 'his', 'she', 'her', 'they', 'their', 'them']

def get_neural_coreference(doc):
    doc = nlp(doc)
    new_doc = []
    for i in range(len(doc)):
        if doc[i:i+1]._.is_coref:
            new_doc.append(doc[i:i+1]._.coref_cluster[0].text)
        else:
            new_doc.append(doc[i:i+1][0].text)
    return [x for x in new_doc if x != '']


def tokenize_spacy(doc):
    s = nlp(doc)
    return [w.text for w in s]


def get_spacy_dependency(sent):
    if len(sent.strip()) == 0:
        return None 
    try:
        sent = nlp(sent)
    except:
        import ipdb; ipdb.set_trace() 

    result = []
    for token in sent:
        word = token.text
        pos = token.pos_
        head = token.head.i 
        if token.dep_ == 'ROOT':
            dependency = 'root'
            head = -1
        else:
            dependency = token.dep_
        result.append({'word': word, 'pos': pos, 'head': head, 'dep': dependency})

    return result 


def get_coref_list(word_dp_list, coref_list):
    word_dp = word_dp_list.copy()
    coref = coref_list.copy()
    word_coref = []
    for i in range(len(word_dp)):
        if i < len(word_dp):
            if word_dp[i] == coref[i]:
                word_coref.append(coref[i])
                continue
            else:
                if i != (len(word_dp)-1):
                    word_dp.pop(i)
                    next_word = word_dp[i]
    #                 print(next_word)
                    next_word_coref_idx = coref.index(next_word)
                    span = ' '.join(coref[i:next_word_coref_idx])
    #                 print(span)
                    word_coref.append(span)
                    for j in range(i,next_word_coref_idx):
                        coref.pop(i)
                    i = i-1
                else:
                    word_coref.append(' '.join(coref[i:]))
    return word_coref


if __name__ == '__main__':
    DATA_DIR = sys.argv[1]
    data_name = sys.argv[2]

    data = pd.read_csv(os.path.join('data', DATA_DIR, f'raw/{data_name}.csv'))
    # data.loc[:, 'sentence'] = data.sentence.str.strip()
    data.loc[:, 'sentence'] = data.sentence.str.strip().replace('\s+', ' ', regex=True)
    data = data.dropna()

    data = data[data.sentence.apply(word_tokenize).apply(len)>5]
    data.to_csv(os.path.join('data', DATA_DIR, 'raw', f'{data_name}.csv'), index=False)

    # dp_data = list(map(get_dependency, data.sentence))
    spacy_dp_data = list(map(get_spacy_dependency, data.sentence))

    # coref_data = list(map(get_coreference, data.sentence))
    neural_coref_data = list(map(get_neural_coreference, data.sentence))

    np.save(os.path.join('data', DATA_DIR, f'interim/coref_{data_name}.npy'), np.array(neural_coref_data, dtype=object))
    # np.save(os.path.join('data', DATA_DIR, f'interim/neural_coref_{data_name}.npy'), np.array(neural_coref_data, dtype=object))

    np.save(os.path.join('data', DATA_DIR, f'interim/dp_{data_name}.npy'), np.array(spacy_dp_data, dtype=object))
    # np.save(os.path.join('data', DATA_DIR, f'interim/spacy_dp_{data_name}.npy'), np.array(spacy_dp_data, dtype=object))    
    
