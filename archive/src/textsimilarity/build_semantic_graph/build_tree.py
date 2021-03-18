"""
In this code, we build a tree for each sentence based on dependency parsing.

There are two crucial parts at this step:
    1st classify each node into two types: 
        'V' for verbs and 'A' for arguments
    2nd classify the child nodes of each node into three groups:
        'verb' for predicates, 'noun' for subjects/objects and 'attribute' for others
"""

import sys
from tqdm import tqdm

import json
import codecs


json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)

verb_pos = ['VBZ', 'VBN', 'VBD', 'VBP', 'VB', 'VBG', 'VERB']
prep_pos = ['PP', 'IN', 'TO']
subj_and_obj = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass'] + ['dobj', 'pobj', 'iobj', 'obj']
conj = ['conj', 'parataxis']
modifier_pos = ['JJ', 'FW', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'ADJ', 'X', 'ADV']
modifiers = ['amod', 'nn', 'mwe', 'advmod', 'quantmod', 'npadvmod', 'advcl', 'poss', 'acl', 'appos',
             'possessive', 'neg', 'auxpass', 'aux', 'det', 'dep', 'predet', 'num','nmod','nummod']


def merge_node(raw, sequence):
    node = {k: v for k, v in raw.items()}
    attribute = raw['attribute']

    attr1, attr2 = [], []   # attr1: ok to merge
    indexes = [idx for idx in node['index']]
    for a in attribute:
        if 'attribute' in a or 'noun' in a or 'verb' in a:
            attr2.append(a)
        elif ((a['dep'] in modifiers or a['pos'] in modifier_pos) and a['pos'] not in prep_pos) | (a['dep']=='compound'):
            attr1.append(a)
            indexes += [idx for idx in a['index']]
        elif (a['dep'] == 'punct') and (a['index'][0] in [idx+1 for idx in indexes]):
            attr1.append(a)
            indexes += [idx for idx in a['index']]
        else:
            attr2.append(a)

    if len(attr1) > 0:
        indexes.sort(key=lambda x:x)
        flags = [index not in indexes[:idx] for idx, index in enumerate(indexes)]
        if len(indexes) == indexes[-1] - indexes[0] + 1 and all(flags):     # need to be consecutive modifiers
            node['word'] = [sequence[i] for i in indexes]
            node['index'] = indexes
            if len(attr2) > 0:
                node['attribute'] = [a for a in attr2]
            else:
                del node['attribute']
        
    
    return node


def build_detailed_tree(sequence, all_dep, root, word_type):
    def is_noun(node):
        return node['dep'] in subj_and_obj or (all_dep[root]['dep'] in subj_and_obj and node['dep'] == 'conj')
    
    def is_verb(node):
        return (node['dep'] == 'cop' and word_type == 'A') or (word_type == 'V' and node['dep'] == 'conj')
    ##=== initialize tree-node ===##
    element = all_dep[root]
    word_type = 'V' if element['pos'] in verb_pos else 'A'
    node = {'word': [sequence[root]], 'index': [root], 'type': word_type, 'dep': element['dep'], 'pos': element['pos']}
    ##=== classify child node sets ===##
    children = [(i, elem) for i, elem in enumerate(all_dep) if elem['head'] == root]
    nouns = [child for child in children if is_noun(child[1])]
    if len(nouns) > 0:
        node['noun'] = [build_detailed_tree(sequence, all_dep, child[0], 'A') for child in nouns]
    verbs = [child for child in children if is_verb(child[1])]
    if len(verbs) > 0:
        node['verb'] = [build_detailed_tree(sequence, all_dep, child[0], 'V') for child in verbs]
    attributes = [child for child in children if child not in nouns + verbs]
    if len(attributes) > 0:
        node['attribute'] = [build_detailed_tree(sequence, all_dep, child[0], 'A') for child in attributes]
    ##=== do node-merging ===##
    if 'attribute' in node:
#         print('running merge node on: ', node['word'])
        node = merge_node(node, sequence)
        
    return node


def build_tree(sent):
    dep, sequence = sent['dependency_parse'], sent['coreference']
    root = [i for i in range(len(dep)) if dep[i]['head'] == -1]
    heads_dep = [w['dep'] for w in dep if w['head'] == root[0]]

    word_type = 'V' if dep[root[0]]['pos'] in verb_pos or 'cop' not in heads_dep else 'A'
    tree = build_detailed_tree(sequence, dep, root[0], word_type)

    return {'words': sequence, 'tree': tree}


