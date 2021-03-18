import numpy as np
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")
import spacy
from spacy.tokens import Token
from textsimilarity.get_w2v_features import get_word2vec_embeddings
from textsimilarity.get_similarity import cosine_similarity


Token.set_extension('srl_arg0', default=None)
Token.set_extension('srl_arg1', default=None)
nlp = spacy.load("en")

def srl(sent):
    doc = nlp(sent)
    words = [token.text for token in doc]
    for i, word in enumerate(doc):
        if word.pos_ == "VERB":
            verb = word.text
            verb_labels = [0 for _ in words]
            verb_labels[i] = 1
            instance = predictor._dataset_reader.text_to_instance(doc, verb_labels)
            output = predictor._model.forward_on_instance(instance)
            tags = output['tags']
    
            if "B-ARG0" in tags:
                start = tags.index("B-ARG0")
                end = max([i for i, x in enumerate(tags) if x == "I-ARG0"] + [start]) + 1
                word._.set("srl_arg0", doc[start:end])
    
            if "B-ARG1" in tags:
                start = tags.index("B-ARG1")
                end = max([i for i, x in enumerate(tags) if x == "I-ARG1"] + [start]) + 1
                word._.set("srl_arg1", doc[start:end])
    res = {}
    for w in doc:
        if w.pos_ == "VERB":
            # print("ARG0:", w._.srl_arg0)
            # print("VERB:", w)
            # print("ARG1:", w._.srl_arg1)
            # print("-----------------")
            res[w] = {'ARG0': w._.srl_arg0, 'ARG1': w._.srl_arg1}
    return res

# def get_w2v_features(sent):
#     MAX_LEN = 100
#     sent = [w for w in sent.split()]
#     w2v_embedding = np.zeros((MAX_LEN, 96))
#     length = len(sent)
#     w2v_embedding[:length] = np.array([nlp(w).vector for w in sent])
#     return w2v_embedding

# def cosine_similarity(vec1, vec2):
#     dot = np.dot(vec1, vec2)
#     norm1 = np.linalg.norm(vec1)
#     norm2 = np.linalg.norm(vec2)
#     return dot/(norm1*norm2)

def get_all_triples(sent):
    parsing_result = srl(sent)
    res = []
    for verb, args in parsing_result.items():
        res.append((args['ARG0'], verb, args['ARG1']))
    return res

def get_first_level_triples(triples):
    if triples == []:
        return []
    
    res = []
    for i, triple in enumerate(triples):
        verb = triple[1]
        first_level_verb_flag = True
        for j, other_triple in enumerate(triples):
            if i == j:
                continue
            other_triple_arg0 = other_triple[0].text if other_triple[0] is not None else ''
            other_triple_arg1 = other_triple[2].text if other_triple[2] is not None else ''
            
            if verb.text in other_triple_arg0 or verb.text in other_triple_arg1:
                # print('{} not in subject: {} and object: {}'.format(verb.text, other_triple[0].text, other_triple[2].text))
                first_level_verb_flag = False
        if first_level_verb_flag == True:
            res.append(triple)
    return res

def get_child_triples(triple, triple_list):
    res = []
    for other_triple in triple_list:
        verb = other_triple[1]
        triple_arg0 = triple[0].text if triple[0] is not None else ''
        triple_arg1 = triple[2].text if triple[2] is not None else ''
        if verb.text in triple_arg0 or verb.text in triple_arg1:
            res.append(other_triple)
    return res

def remove_triple(triple, triple_list, matched):
    if matched == False:
        # for unmatched triple, just remove the triple
        triple_list.remove(triple)
    else:
        # for matched triple, remove the triple and all child triples
        child_triples = get_child_triples(triple, triple_list)
        for child_triple in child_triples:
            triple_list.remove(child_triple)
        triple_list.remove(triple)
        
def get_wms_srl_recursive(sent1, sent2):
    # print(sent1)
    # print(sent2)
    verb_threshold = 0.5
    sent1 = sent1.replace(',', ' ')
    sent1 = sent1.replace('.', ' ')
    sent1 = sent1.replace('"', ' ')
    sent1 = sent1.replace("'", ' ')
    sent1 = sent1.replace("’", ' ')
    sent1 = sent1.lower()
    sent2 = sent2.replace(',', ' ')
    sent2 = sent2.replace('.', ' ')
    sent2 = sent1.replace('"', ' ')
    sent2 = sent1.replace("'", ' ')
    sent2 = sent2.replace("’", ' ')
    sent2 = sent2.lower()
    # we treat sent1 as the query sentence and sent2 as the candidate sentence
    all_triples1 = get_all_triples(sent1)
    all_triples2 = get_all_triples(sent2)
    
    first_level_triples1 = get_first_level_triples(all_triples1)
    first_level_triples2 = get_first_level_triples(all_triples2)
    
    doc1 = [w for w in sent1.split()]
    doc2 = [w for w in sent2.split()]
    
    emb1 = get_word2vec_embeddings(sent1)
    emb2 = get_word2vec_embeddings(sent2)
    
    # similarity is 0 if no pair of verbs is similar
    sim_sent = 0
    
    if all_triples1 == []:
        # print('SRL got no result on sentence: {}'.format(sent1))
        return 0
    elif all_triples2 == []:
        # print('SRL got no result on sentence: {}'.format(sent2))
        return 0
    
    while all_triples2:
        for triple2 in first_level_triples2:
            # find the the most similar verb in sent1
            v2 = triple2[1].text
            max_sim_verb = 0
            nearest_triple1 = None
            for triple1 in first_level_triples1:
                v1 = triple1[1].text
                sim_verb = cosine_similarity(nlp(v2).vector, nlp(v1).vector)
                if sim_verb >= max_sim_verb:
                    max_sim_verb = sim_verb
                    nearest_triple1 = triple1
            
            # print('The most similar verb for {} is {}, similarity: {}.'.format(v2, nearest_triple1[1].text, max_sim_verb))
            
            if nearest_triple1 is None:
                sim_triple = 0
                remove_triple(triple2, all_triples2, False)
                
            # compute triple similarity
            elif max_sim_verb < verb_threshold:
                # unmatched triples
                # print('Verb similarity smaller than threshold: {}, triple similarity is zero.'.format(verb_threshold))
                sim_triple = 0
                
                # remove the triples
                remove_triple(nearest_triple1, all_triples1, False)
                remove_triple(triple2, all_triples2, False)
                
            else:
                # matched triples
                sim_verb = cosine_similarity(nlp(triple2[1].text).vector, nlp(nearest_triple1[1].text).vector)
                
                # calculate the similarity in ARG0
                if triple2[0] is None or nearest_triple1[0] is None:
                    sim_arg0 = 0
                else:
                    arg0_1 = nearest_triple1[0].text
                    arg0_2 = triple2[0].text
                    arg0_1 = [w for w in arg0_1.split()]
                    arg0_2 = [w for w in arg0_2.split()]
                    arg0_1_emb = [emb1[doc1.index(w)] for w in arg0_1]
                    arg0_2_emb = [emb2[doc2.index(w)] for w in arg0_2]
                    s1 = set(arg0_1)
                    s2 = set(arg0_2)
                    d2 = {}
                    for s in s2:
                        d2[s] = arg0_2.count(s)/len(arg0_2)
                    arg0_sim_list = []
                    for i2, e2 in enumerate(arg0_2_emb):
                        word_sim = []
                        for e1 in arg0_1_emb:
                            word_sim.append(cosine_similarity(e1, e2))
                        arg0_sim_list.append(max(word_sim)*d2[arg0_2[i2]])
                    sim_arg0 = sum(arg0_sim_list)
                # print('ARG0 similarity for verb {}: {}'.format(v2, sim_arg0))
                
                # calculate similarity in ARG1
                if triple2[2] is None or nearest_triple1[2] is None:
                    sim_arg1 = 0
                else:
                    arg1_1 = nearest_triple1[2].text
                    arg1_2 = triple2[2].text
                    arg1_1 = [w for w in arg1_1.split()]
                    arg1_2 = [w for w in arg1_2.split()]
                    arg1_1_emb = [emb1[doc1.index(w)] for w in arg1_1]
                    arg1_2_emb = [emb2[doc2.index(w)] for w in arg1_2]
                    s1 = set(arg1_1)
                    s2 = set(arg1_2)
                    d2 = {}
                    for s in s2:
                        d2[s] = arg1_2.count(s)/len(arg1_2)
                    arg1_sim_list = []
                    for i2, e2 in enumerate(arg1_2_emb):
                        word_sim = []
                        for e1 in arg1_1_emb:
                            word_sim.append(cosine_similarity(e1, e2))
                        arg1_sim_list.append(max(word_sim)*d2[arg1_2[i2]])
                    sim_arg1 = sum(arg1_sim_list)
                # print('ARG1 similarity for verb {}: {}'.format(v2, sim_arg1))
                
                sim_triple = sim_verb + sim_arg0 + sim_arg1
                
                # remove the triples and child triples
                remove_triple(nearest_triple1, all_triples1, True)
                remove_triple(triple2, all_triples2, True)
            
            sim_sent += sim_triple
            first_level_triples1 = get_first_level_triples(all_triples1)
            first_level_triples2 = get_first_level_triples(all_triples2)
    
    return sim_sent

