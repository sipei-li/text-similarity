import sys
from tqdm import tqdm
import os
#from build_tree import build_tree
#from prune_and_merge_tree import prune
#from rearrange_tree import rearrange
#from build_graph import get_graph
#from merge_graph import merge

from textsimilarity.build_semantic_graph.build_tree import build_tree
from textsimilarity.build_semantic_graph.prune_and_merge_tree import prune
from textsimilarity.build_semantic_graph.rearrange_tree import rearrange
from textsimilarity.build_semantic_graph.build_graph import get_graph
from textsimilarity.build_semantic_graph.merge_graph import merge
from textsimilarity.preprocess import get_coref_and_dp
from textsimilarity import get_w2v_features

import numpy as np
import codecs

def merge_dp_coref(dp, coref):
    sents = []
    for i in range(len(dp)):
        sent = {'dependency_parse':dp[i], 'coreference':coref[i]}
        sents.append(sent)
    return sents


def get_graph_from_sent(sentence):
    dp = get_coref_and_dp.get_spacy_dependency(sentence)
    coref = get_coref_and_dp.get_neural_coreference(sentence)
    data = merge_dp_coref([dp], [coref])
    sent = build_tree(data[0])
    sent = {'sequence':sent['words'], 'tree':prune(sent['tree'], sent['words'])}
    sent = {'sequence': sent['sequence'], 'tree': rearrange(sent['tree'], sent['sequence'])}
    return get_graph(sent['tree']), dp, coref


if __name__ == '__main__':
    # SST_INTERIM_DIR = 'data/twitter/interim'
    # SST_PROCESSED_DIR = 'data/twitter/processed'
    DATA_DIR = sys.argv[1]
    data_name = sys.argv[2]

    # coref_train = np.load(os.path.join(SST_INTERIM_DIR, 'coref_train.npy'),allow_pickle=True)
    # coref_val = np.load(os.path.join(SST_INTERIM_DIR, 'coref_dev.npy'),allow_pickle=True)
    # coref_test = np.load(os.path.join(SST_INTERIM_DIR, 'coref_test.npy'),allow_pickle=True)
    coref_data = np.load(os.path.join('data', DATA_DIR, f'interim/coref_{data_name}.npy'), allow_pickle=True)

    # dp_train = np.load(os.path.join(SST_INTERIM_DIR, 'dp_train.npy'),allow_pickle=True)
    # dp_val = np.load(os.path.join(SST_INTERIM_DIR, 'dp_dev.npy'),allow_pickle=True)
    # dp_test = np.load(os.path.join(SST_INTERIM_DIR, 'dp_test.npy'),allow_pickle=True)
    dp_data = np.load(os.path.join('data', DATA_DIR, f'interim/dp_{data_name}.npy'), allow_pickle=True)

    # sents_train = merge_dp_coref(dp_train, coref_train)
    # sents_val = merge_dp_coref(dp_val, coref_val)
    # sents_test = merge_dp_coref(dp_test, coref_test)
    sents_data = merge_dp_coref(dp_data, coref_data)

    # graphs_train = []
    # for sent in sents_train:
    #     sent = build_tree(sent)
    #     sent = {'sequence':sent['words'], 'tree':prune(sent['tree'], sent['words'])}
    #     sent = {'sequence': sent['sequence'], 'tree': rearrange(sent['tree'], sent['sequence'])}
    #     graph = get_graph(sent['tree'])
    #     graphs_train.append(graph)

    # graphs_val = []
    # for sent in sents_val:
    #     sent = build_tree(sent)
    #     sent = {'sequence':sent['words'], 'tree':prune(sent['tree'], sent['words'])}
    #     sent = {'sequence': sent['sequence'], 'tree': rearrange(sent['tree'], sent['sequence'])}
    #     graph = get_graph(sent['tree'])
    #     graphs_val.append(graph)

    # graphs_test = []
    # for sent in sents_val:
    #     sent = build_tree(sent)
    #     sent = {'sequence':sent['words'], 'tree':prune(sent['tree'], sent['words'])}
    #     sent = {'sequence': sent['sequence'], 'tree': rearrange(sent['tree'], sent['sequence'])}
    #     graph = get_graph(sent['tree'])
    #     graphs_test.append(graph)

    graphs_data = []
    for sent in sents_data:
        sent = build_tree(sent)
        sent = {'sequence':sent['words'], 'tree':prune(sent['tree'], sent['words'])}
        sent = {'sequence': sent['sequence'], 'tree': rearrange(sent['tree'], sent['sequence'])}
        graph = get_graph(sent['tree'])
        graphs_data.append(graph)

    
    # np.save(os.path.join(SST_PROCESSED_DIR, 'graphs_train.npy'), np.array(graphs_train, dtype=object))
    # np.save(os.path.join(SST_PROCESSED_DIR, 'graphs_dev.npy'), np.array(graphs_val, dtype=object))
    # np.save(os.path.join(SST_PROCESSED_DIR, 'graphs_test.npy'), np.array(graphs_test, dtype=object))
    np.save(os.path.join('data', DATA_DIR, f'processed/graphs_{data_name}.npy'), np.array(graphs_data, dtype=object))
    