import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# graph = np.load('../../data/twitter/graphs_twitter.npy')[0]

def draw_graph(graph):
    out = nx.DiGraph()
    
    for node in graph['nodes']:
        out.add_node(node['word'])
    
    labels = {}
    for i in range(len(graph['edges'])):
        node1_edge = graph['edges'][i]
        for j in range(len(node1_edge)):
            if node1_edge[j] != '':
                out.add_edge(graph['nodes'][i]['word'], graph['nodes'][j]['word'])
                labels[(graph['nodes'][i]['word'], graph['nodes'][j]['word'])] = node1_edge[j]

    pos_ = nx.spring_layout(out)
    plt.figure(figsize=(10,10))
    nx.draw(out, pos_)
    nx.draw_networkx_nodes(out, pos_)
    nx.draw_networkx_labels(out, pos_)
    nx.draw_networkx_edge_labels(out, pos_, edge_labels=labels)