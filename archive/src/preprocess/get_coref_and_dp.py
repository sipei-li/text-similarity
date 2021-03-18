import numpy as np
import pandas as pd
import os
# import nltk
import sys
import codecs
import json
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
import allennlp_models.rc
from nltk import word_tokenize
import spacy
import neuralcoref 

nlp = spacy.load("en")

# Add neural coref to SpaCy's pipe
neuralcoref.add_to_pipe(nlp)

dependency_parser = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
coref_reslt = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")

pronouns = ['it', 'its', 'he', 'him', 'his', 'she', 'her', 'they', 'their', 'them']

def get_coreference(doc):

    def get_crf(span, words):
        phrase = []
        for i in range(span[0], span[1] + 1):
            phrase += [words[i]]
        return (' '.join(phrase), span[0], span[1] - span[0] + 1)

    def get_best(crf):
        crf.sort(key=lambda x: x[2], reverse=True)
        if crf[0][2] == 1:
            crf.sort(key=lambda x: len(x[0]), reverse=True)
        for w in crf:
            if w[0].lower() not in pronouns and w[0].lower() != '\t':
                return w[0]
        return None

    doc = coref_reslt.predict(document=doc)
    words = [w.strip(' ') for w in doc['document']]
    clusters = doc['clusters']

    if clusters == []:
        entity = ' '.join(words)
    else:
        for group in clusters:
            crf = [get_crf(span, words) for span in group]
            entity = get_best(crf)
            if entity in ['\t', None]:
                entity = ' '.join(words)
            if entity not in ['\t', None]:
                for phrase in crf:
                    if phrase[0].lower() in pronouns:
                        index = phrase[1]
                        words[index] = entity

    doc, sent = [], []
    for word in words:
        if word.strip(' ') == '\t':
            doc.append(sent)
            sent = []
        else:
            if word.count('\t'):
                print(word)
                word = word.strip('\t')
            sent.append(word)
    doc.append(sent)
    if len(doc)==1:
        doc = doc[0]
    return [x for x in doc if x != '']

def get_neural_coreference(doc):
    doc = nlp(doc)
    new_doc = []
    for i in range(len(doc)):
        if doc[i:i+1]._.is_coref:
            new_doc.append(doc[i:i+1]._.coref_cluster[0].text)
        else:
            new_doc.append(doc[i:i+1][0].text)
    return [x for x in new_doc if x != '']

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


def get_dependency(sent):
    if len(sent.strip()) == 0:
        return None
    try:
        sent = dependency_parser.predict(sentence=sent)
    except:
        import ipdb; ipdb.set_trace()
    words, pos, heads, dependencies = sent['words'], sent['pos'], sent['predicted_heads'], sent['predicted_dependencies']
    result = [{'word':w, 'pos':p, 'head':h - 1, 'dep':d} for w, p, h, d in zip(words, pos, heads, dependencies)]
    return result


def dependency_parse(raw):
    context = {
        key: [
            get_dependency(sent, dependency_parser) for sent in value
        ] for key, value in tqdm(raw.items(), desc='   - (Dependency Parsing: 1st) -   ')
    }
    return context


def tokenize_spacy(doc):
    s = nlp(doc)
    return [w.text for w in s]


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

if __name__ == '__main__':
    DATA_DIR = sys.argv[1]
    data_name = sys.argv[2]

    data = pd.read_csv(os.path.join('data', DATA_DIR, f'raw/{data_name}.csv'))
    # data.loc[:, 'sentence'] = data.sentence.str.strip()
    data.loc[:, 'sentence'] = data.sentence.str.strip().replace('\s+', ' ', regex=True)


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
