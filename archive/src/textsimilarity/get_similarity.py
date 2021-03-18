import sys 
import numpy as np
import os
from transformers import BertModel, BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from nltk.corpus import stopwords
import nltk
from textsimilarity.build_semantic_graph import build_semantic_graph
nltk.download('stopwords')

stop_words = set(stopwords.words('english')) 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab = tokenizer.get_vocab()
vocab_list = list(vocab.keys())


def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot/(norm1*norm2)


def get_wms(doc1, doc2, emb1, emb2):
    # we treat doc1 as the query document
    result1 = [(word, emb) for word, emb in zip(doc1,emb1) if (word not in stop_words) and (not np.all(emb==0))]
    document1 = [x[0] for x in result1]
    embedding1 = [x[1] for x in result1]
    
    result2 = [(word, emb) for word, emb in zip(doc2,emb2) if (word not in stop_words) and (not np.all(emb==0))]
    document2 = [x[0] for x in result2]
    embedding2 = [x[1] for x in result2]

    if len(embedding1)==0 or len(embedding2)==0:
        return 0
    else:
        s1 = set(document1)
        s2 = set(document2)
        
        d2 = {}
        for s in s2:
            d2[s] = document2.count(s)/len(document2)
        
        sent_sim = []
        for i2, e2 in enumerate(embedding2):
            word_sim = []
            for e1 in embedding1:
                word_sim.append(cosine_similarity(e1, e2))
            # print(document2[i2], word_sim)
            sent_sim.append(max(word_sim)*d2[document2[i2]])
        return sum(sent_sim)

def get_node_embeddings(graph, embeddings):
    node_features = []
    for node in graph['nodes']:
        node_features.append(embeddings[[node['index']]].mean(axis=0))            
    return np.array(node_features)


def get_graph_similarity(graph1, graph2, emb1, emb2, threshold):
    # we treat graph1 as the query document
    embedding1 = get_node_embeddings(graph1, emb1)
    embedding2 = get_node_embeddings(graph2, emb2)
    
    w_match = 1
    w_nonmatch = 0.5
    
    graph_sim = []
    node_match = []
    for i2, e2 in enumerate(embedding2):
        node_sim = []
        for e1 in embedding1:
            if (np.all(e1==0))|(np.all(e2==0)):
                node_sim.append(0)
            else:
                node_sim.append(cosine_similarity(e1, e2))
        max_sim_node = np.argmax(node_sim)
        if max(node_sim) >= threshold:
            node_match.append((max_sim_node, i2))
        graph_sim.append(max(node_sim))
        # print(f"{graph2['nodes'][i2]['word']} matches {graph1['nodes'][max_sim_node]['word']}")
        
#     print('node match: ', node_match)
    if len(node_match) <= 1:
        return sum(graph_sim)*w_nonmatch/len(graph2), None
    else:
        weights = np.zeros(len(graph_sim)) + w_nonmatch
        for i in range(len(node_match)):
            for j in range(i, len(node_match)):
                n1_g1 = node_match[i][0]
                n1_g2 = node_match[i][1]
                n2_g1 = node_match[j][0]
                n2_g2 = node_match[j][1]
                
                n1_n2_g1 = graph1['edges'][n1_g1][n2_g1]
                n2_n1_g1 = graph1['edges'][n2_g1][n1_g1]

                n1_n2_g2 = graph2['edges'][n1_g2][n2_g2]
                n2_n1_g2 = graph2['edges'][n2_g2][n1_g2]

                if ((n1_n2_g1!='')&(n1_n2_g1 == n1_n2_g2))|((n2_n1_g1!='')&(n2_n1_g1 == n2_n1_g2)):
                    weights[n1_g2] += (w_match-w_nonmatch)
                    weights[n2_g2] += (w_match-w_nonmatch)
#                     print(n1_g2)
#                     print(n2_g2)
                    # print('(', graph1['nodes'][n1_g1]['word'], ',', graph1['nodes'][n2_g1]['word'], ') matches ','(',
                    #      graph2['nodes'][n1_g2]['word'], ',', graph2['nodes'][n2_g2]['word'], ')')
                    
        return np.dot(graph_sim, weights)/len(graph2), weights


def get_graph_similarity_wms(graph1, graph2, emb1, emb2, threshold):
    
    # we treat graph1 as the query document
    embedding1 = get_node_embeddings(graph1, emb1)
    embedding2 = get_node_embeddings(graph2, emb2)
    
    w_match = 1
    w_nonmatch = 0.5
    
    graph_sim = []
    node_match = []
    for i2, n2 in enumerate(graph2['nodes']):
        node_sim = []
        for i1, n1 in enumerate(graph1['nodes']):
            sent_wms = get_wms(n1['word'].split(' '), 
                                   n2['word'].split(' '),
                                   emb1.take(n1['index'], axis=0),
                                   emb2.take(n2['index'], axis=0))
            if sent_wms == None:
                node_sim.append(0)
            else:
                node_sim.append(sent_wms)
        max_sim_node = np.argmax(node_sim)
        if max(node_sim) >= threshold:
            node_match.append((max_sim_node, i2))
        graph_sim.append(max(node_sim))
        # print(f"{graph2['nodes'][i2]['word']} matches {graph1['nodes'][max_sim_node]['word']}")
        
    # print(np.array(graph_sim)/len(graph2))
#     print('node match: ', node_match)
    if len(node_match) <= 1:
        return sum(graph_sim)*w_nonmatch/len(graph2), None
    else:
        weights = np.zeros(len(graph_sim)) + w_nonmatch
        for i in range(len(node_match)):
            for j in range(i, len(node_match)):
                n1_g1 = node_match[i][0]
                n1_g2 = node_match[i][1]
                n2_g1 = node_match[j][0]
                n2_g2 = node_match[j][1]
                
                n1_n2_g1 = graph1['edges'][n1_g1][n2_g1]
                n2_n1_g1 = graph1['edges'][n2_g1][n1_g1]

                n1_n2_g2 = graph2['edges'][n1_g2][n2_g2]
                n2_n1_g2 = graph2['edges'][n2_g2][n1_g2]

                if ((n1_n2_g1!='')&(n1_n2_g1 == n1_n2_g2))|((n2_n1_g1!='')&(n2_n1_g1 == n2_n1_g2)):
                    weights[n1_g2] += (w_match-w_nonmatch)/len(graph2)
                    weights[n2_g2] += (w_match-w_nonmatch)/len(graph2)
#                     print(n1_g2)
#                     print(n2_g2)
                    # print('(', graph2['nodes'][n1_g2]['word'], ',', graph2['nodes'][n2_g2]['word'], ') matches ','(',
                    #      graph1['nodes'][n1_g1]['word'], ',', graph1['nodes'][n2_g1]['word'], ')')
                    
        return np.dot(graph_sim, weights)/len(graph2), weights


def get_sent_similarity(coref1, coref2, w2v1, w2v2, g1=None, g2=None, wms_weight=1, sim_method='word2vec', threshold=None):
    # g1, w2v1, dp1, coref1 = build_semantic_graph.get_graph_from_sent(sent1)
    # g2, w2v2, dp2, coref2 = build_semantic_graph.get_graph_from_sent(sent2)
    
    if wms_weight == 0:
        if sim_method=='word2vec':
            graph_sim = get_graph_similarity(g1, g2, w2v1, w2v2, threshold=threshold)[0]
        elif sim_method=='wms':
            graph_sim = get_graph_similarity_wms(g1, g2, w2v1, w2v2, threshold=threshold)[0]
        return graph_sim
    elif wms_weight == 1:
        wms = get_wms(coref1, coref2, w2v1, w2v2)
        return wms
    else:
        wms = get_wms(coref1, coref2, w2v1, w2v2)
        if sim_method=='word2vec':
            graph_sim = get_graph_similarity(g1, g2, w2v1, w2v2, threshold=threshold)[0]
        elif sim_method=='wms':
            graph_sim = get_graph_similarity_wms(g1, g2, w2v1, w2v2, threshold=threshold)[0]
        return wms_weight*wms + (1-wms_weight)*graph_sim


if __name__ == '__main__':
    # for train-test splited data
    
    DATA_DIR = sys.argv[1]
    THRESHOLD=float(sys.argv[2])
    WMS_WEIGHT = float(sys.argv[3])
    SIM_METHOD=sys.argv[4]

    coref_train = np.load(os.path.join('data', DATA_DIR, 'interim', f'coref_train.npy'), allow_pickle=True)
    coref_train = list(map(lambda x: [j for n in x for j in n.split()], coref_train))

    coref_test = np.load(os.path.join('data', DATA_DIR, 'interim', f'coref_test.npy'), allow_pickle=True)
    coref_test = list(map(lambda x: [j for n in x for j in n.split()], coref_test))

    graph_train = np.load(os.path.join('data', DATA_DIR, 'processed', f'graphs_train.npy'), allow_pickle=True)
    graph_test = np.load(os.path.join('data', DATA_DIR, 'processed', f'graphs_test.npy'), allow_pickle=True)
    
    w2v_embeddings_train = np.load(os.path.join('data', DATA_DIR, 'interim', f'word2vec_embeddings_train.npy'), allow_pickle=True)
    w2v_embeddings_test = np.load(os.path.join('data', DATA_DIR, 'interim', f'word2vec_embeddings_test.npy'), allow_pickle=True)

    sim_train = np.zeros((w2v_embeddings_train.shape[0], w2v_embeddings_train.shape[0]))
    num_edge_match = 0
    print(f'*************start calculating training data wms weight {WMS_WEIGHT} **************')
    for i in range(sim_train.shape[0]):
        for j in range(i+1, sim_train.shape[0]):
            sim = get_sent_similarity(coref_train[i], coref_train[j], 
                                    w2v_embeddings_train[i], w2v_embeddings_train[j],
                                    graph_train[i], graph_train[j], 
                                    WMS_WEIGHT, SIM_METHOD, THRESHOLD)
            if type(sim) == tuple:
                sim_train[i,j] = sim[0]
                if max(sim[1]) > 0.5:
                    num_edge_match += 1
            else:
                sim_train[i,j] = sim
            if np.isnan(sim_train[i,j]):
                    print('j: ', j)
        print(i)
    print('*************finished training data:**************')
    sim_train = sim_train + sim_train.T + np.identity(sim_train.shape[0])*sim_train.max()
    np.save(os.path.join('data', DATA_DIR, 'processed', f'sim_train_{SIM_METHOD}_wmsweight_{WMS_WEIGHT}_threshold{THRESHOLD}.npy'), sim_train)
    np.save(os.path.join('data', DATA_DIR, 'processed', f'dist_train_{SIM_METHOD}_wmsweight_{WMS_WEIGHT}_threshold{THRESHOLD}.npy'), sim_train.max()-sim_train)
    

    sim_test = np.zeros((w2v_embeddings_test.shape[0], w2v_embeddings_train.shape[0]))
    num_edge_match_test = 0
    print('*************start calculating testing data**************')
    for i in range(sim_test.shape[0]):
        for j in range(w2v_embeddings_train.shape[0]):
            sim = get_sent_similarity(coref_train[j], coref_test[i], 
                                    w2v_embeddings_train[j], w2v_embeddings_test[i],
                                    graph_train[j], graph_test[i],  
                                    WMS_WEIGHT, SIM_METHOD, THRESHOLD)
            if type(sim) == tuple:
                sim_test[i,j] = sim[0]
                if max(sim[1]) > 0.5:
                    num_edge_match_test += 1
            else:
                sim_test[i,j] = sim
        print(i)
    print('*************finished testing data:**************')
    np.save(os.path.join('data', DATA_DIR, 'processed', f'sim_test_{SIM_METHOD}_wmsweight_{WMS_WEIGHT}_threshold{THRESHOLD}.npy'), sim_test)
    np.save(os.path.join('data', DATA_DIR, 'processed', f'dist_test_{SIM_METHOD}_wmsweight_{WMS_WEIGHT}_threshold{THRESHOLD}.npy'), sim_train.max()-sim_test)

    # print('*************start calculating wms training data**************')
    # sim_train = np.zeros((w2v_embeddings_train.shape[0], w2v_embeddings_train.shape[0]))
    # for i in range(sim_train.shape[0]):
    #     for j in range(i+1, sim_train.shape[0]):
    #         sim_train[i,j] = get_wms(coref_train[i], coref_train[j], 
    #                         w2v_embeddings_train[i], w2v_embeddings_train[j])
    #     print(i)
    # print('*************finished wms training data:**************')
    # sim_train = sim_train + sim_train.T + np.identity(sim_train.shape[0])*sim_train.max()
    # np.save(os.path.join('data', DATA_DIR, 'processed', 'wms_train.npy'), sim_train)
    # np.save(os.path.join('data', DATA_DIR, 'processed', 'wmd_train.npy'), sim_train.max()-sim_train)

    # print('*************start calculating wms testing data**************')
    # sim_test = np.zeros((w2v_embeddings_test.shape[0], w2v_embeddings_train.shape[0]))
    # for i in range(sim_test.shape[0]):
    #     for j in range(w2v_embeddings_train.shape[0]):
    #         sim_test[i,j] = get_wms(coref_train[j], coref_test[i], 
    #                         w2v_embeddings_train[j], w2v_embeddings_test[i])
    #     print(i)
    # print('*************finished wms testing data:**************')
    # np.save(os.path.join('data', DATA_DIR, 'processed', 'wms_test.npy'), sim_test)
    # np.save(os.path.join('data', DATA_DIR, 'processed', 'wmd_test.npy'), sim_train.max()-sim_test)


    # threshold=float(sys.argv[2])
    # g_sim_train = np.zeros((w2v_embeddings_train.shape[0], w2v_embeddings_train.shape[0]))
    # num_edge_match = 0
    # print('*************start calculating graph training data**************')
    # for i in range(g_sim_train.shape[0]):
    #     for j in range(i+1, g_sim_train.shape[0]):
    #         sim = get_graph_similarity(graph_train[i], graph_train[j], 
    #                         w2v_embeddings_train[i], w2v_embeddings_train[j], THRESHOLD)
    # #             if np.count_nonzero(np.isnan(sim))
    #         if type(sim) == tuple:
    #             g_sim_train[i,j] = sim[0]
    #             if max(sim[1]) > 0.5:
    #                 num_edge_match += 1
    #         else:
    #             g_sim_train[i,j] = sim
    #         if np.isnan(g_sim_train[i,j]):
    #                 print('j: ', j)
    #     print(i)
    # print('*************finished graph training data:**************')
    # g_sim_train = g_sim_train + sim_train.T + np.identity(sim_train.shape[0])*g_sim_train.max()
    # np.save(os.path.join('data', DATA_DIR, 'processed', f'sim_train_word2vec_threshold{THRESHOLD}.npy'), g_sim_train)
    # np.save(os.path.join('data', DATA_DIR, 'processed', f'dist_train_word2vec_threshold{THRESHOLD}.npy'), g_sim_train.max()-g_sim_train)
    

    # g_sim_test = np.zeros((w2v_embeddings_test.shape[0], w2v_embeddings_train.shape[0]))
    # num_edge_match_test = 0
    # print('*************start calculating graph testing data**************')
    # for i in range(g_sim_test.shape[0]):
    #     for j in range(w2v_embeddings_train.shape[0]):
    #         sim = get_graph_similarity(graph_train[j], graph_test[i], 
    #                         w2v_embeddings_train[j], w2v_embeddings_test[i], threshold)
    #         if type(sim) == tuple:
    #             g_sim_test[i,j] = sim[0]
    #             if max(sim[1]) > 0.5:
    #                 num_edge_match_test += 1
    #         else:
    #             g_sim_test[i,j] = sim
    #     print(i)
    # print('*************finished graph testing data:**************')
    # np.save(os.path.join('data', DATA_DIR, 'processed', f'g_sim_test_word2vec_threshold{threshold}.npy'), g_sim_test)
    # np.save(os.path.join('data', DATA_DIR, 'processed', f'g_dist_test_word2vec_threshold{threshold}.npy'), g_sim_train.max()-g_sim_test)

    """
    # for not splited data
    DATA_DIR = sys.argv[1]
    data_name = sys.argv[2]
    
    # get coreference
    coref_data = np.load(os.path.join('data', DATA_DIR, 'interim', f'coref_{data_name}.npy'), allow_pickle=True)
    coref_data = list(map(lambda x: [j for n in x for j in n.split()], coref_data))
    
    # choose an embedding: bert / word2vec
    # bert_embeddings_data = np.load(os.path.join('data', DATA_DIR, 'interim', f'bert_embeddings_{data_name}.npy'))
    word2vec_embeddings_data = np.load(os.path.join('data', DATA_DIR, 'interim', f'word2vec_embeddings_{data_name}.npy'), allow_pickle=True)

    # compute word mover's similarity
    sim_data = np.zeros((word2vec_embeddings_data.shape[0], word2vec_embeddings_data.shape[0]))
    for i in range(sim_data.shape[0]):
        for j in range(i+1, sim_data.shape[0]):
            sim_data[i,j] = get_wms(coref_data[i], coref_data[j], 
                            word2vec_embeddings_data[i], word2vec_embeddings_data[j], vocab_list)
        
    sim_data = sim_data + sim_data.T
    np.save(os.path.join('data', DATA_DIR, 'processed', f'wms_{data_name}.npy'), sim_data)
    np.save(os.path.join('data', DATA_DIR, 'processed', f'wmd_{data_name}.npy'), 1-sim_data)

    # get the graphs
    graphs_data = np.load(os.path.join('data', DATA_DIR, f'graphs_{data_name}.npy'), allow_pickle=True)

    # compute the graph similarity (threshold = 0.0)
    g_sim_data_00 = np.zeros((word2vec_embeddings_data.shape[0], word2vec_embeddings_data.shape[0]))
    num_edge_match = 0
    for i in range(g_sim_data_00.shape[0]):
        for j in range(i+1, g_sim_data_00.shape[0]):
            sim = get_graph_similarity(graphs_data[i], graphs_data[j], 
                               word2vec_embeddings_data[i], word2vec_embeddings_data[j], 0)
            if type(sim) == tuple:
                g_sim_data_00[i,j] = sim[0]
                if max(sim[1]) > 0.5:
                    num_edge_match += 1
            else:
                g_sim_data_00[i,j] = sim
    
    # edge_matches.append(num_edge_match)
    g_sim_data_00 = g_sim_data_00 + g_sim_data_00.T + np.identity(g_sim_data_00.shape[0])

    np.save(os.path.join('data', DATA_DIR, 'processed', f'graph_similarity_{data_name}_00.npy'), g_sim_data_00)

    # compute the graph similarity (threshold = 0.9)
    g_sim_data_09 = np.zeros((word2vec_embeddings_data.shape[0], word2vec_embeddings_data.shape[0]))
    num_edge_match = 0
    for i in range(g_sim_data_09.shape[0]):
        for j in range(i+1, g_sim_data_09.shape[0]):
            sim = get_graph_similarity(graphs_data[i], graphs_data[j], 
                               word2vec_embeddings_data[i], word2vec_embeddings_data[j], 0)
            if type(sim) == tuple:
                g_sim_data_09[i,j] = sim[0]
                if max(sim[1]) > 0.5:
                    num_edge_match += 1
            else:
                g_sim_data_09[i,j] = sim
    
    # edge_matches.append(num_edge_match)
    g_sim_data_09 = g_sim_data_09 + g_sim_data_09.T + np.identity(g_sim_data_09.shape[0])

    np.save(os.path.join('data', DATA_DIR, 'processed', f'graph_similarity_{data_name}_09.npy'), g_sim_data_09)
    """

    