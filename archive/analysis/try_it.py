# Import packages for data cleaning
import numpy as np
import pandas as pd
# Import packages for data visualization
import networkx as nx
import matplotlib.pyplot as plt
graph = np.load('../data/obama/graphs_obama.npy', allow_pickle=True)
twitter = nx.DiGraph()

# add nodes into networkx
def add_nodes(i):
    nodes = []
    count = 0
    Y = []
    ind = count
    column = 0
    for node in graph[i]['nodes']:
        Y.append(node['index'][0])
    Y = sorted(Y)
    for y in Y:
        for node in graph[i]['nodes']:
            if y == node['index'][0]:
                twitter.add_node(node['word'], index=count, sentence=i, pos=node['index'][0])
                nodes.append(node['word'])
                count += 1
                break
#add index
add_nodes(0)
add_nodes(1)

'''
for node in graph[0]['nodes']:
    twitter.add_node(node['word'],index=count,sentence = 0,pos = node['index'][0])
    #twitter.add_node(node['word'])
    nodes.append(node['word'])
    count += 1
    #ind = str(column)+'.'+str(count)
count = 0
for node in graph[1]['nodes']:
    twitter.add_node(node['word'],index=count,sentence = 1,pos = node['index'][0])
    #twitter.add_node(node['word'])
    nodes.append(node['word'])
    count += 1

'''

def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center

def rescale_layout(pos, scale=1):
    """Returns scaled position array to (-scale, scale) in all axes.

    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.

    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).

    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.

    See Also
    --------
    rescale_layout_dict
    """
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos


def multiline_layout(G, subset_key="subset", align="vertical", scale=1, center=None):
    """Position nodes in layers of straight lines.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    subset_key : string (default='subset')
        Key of node data to be used as layer subset.

    align : string (default='vertical')
        The alignment of nodes. Vertical or horizontal.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    num_sen: sentences number

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.complete_multipartite_graph(28, 16, 10)
    >>> pos = nx.multipartite_layout(G)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    Network does not need to be a complete multipartite graph. As long as nodes
    have subset_key data, they will be placed in the corresponding layers.

    """

    import numpy as np

    G, center = _process_params(G, center=center, dim=2)
    if len(G) == 0:
        return {}

    layers = {}
    for v, data in G.nodes(data=True):
        try:
            layer = data[subset_key]
            sent = data['sentence']
        except KeyError:
            msg = "all nodes must have subset_key (default='subset') as data"
            raise ValueError(msg)
        layers[layer] = [v] + layers.get(layer, [])

    pos = None
    nodes = []
    if align == "vertical":
        width = len(layers)
        for i, layer in layers.items():
            height = len(layer)
            xs = np.repeat(i, height)
            ys = np.arange(0, height, dtype=float)
            #ys = np.arange(0, height, dtype=float)
            offset = ((width - 1) / 2, (height - 1) / 2)
            layer_pos = np.column_stack([xs, ys]) - offset
            if pos is None:
                pos = layer_pos
            else:
                pos = np.concatenate([pos, layer_pos])
            nodes.extend(layer)
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(nodes, pos))
        return pos

    if align == "horizontal":
        height = len(layers)
        for i, layer in layers.items():
            width = len(layer)
            xs = np.arange(0, width, dtype=float)
            ys = np.repeat(i, width)
            offset = ((width - 1) / 2, (height - 1) / 2)
            layer_pos = np.column_stack([xs, ys]) - offset
            if pos is None:
                pos = layer_pos
            else:
                pos = np.concatenate([pos, layer_pos])
            nodes.extend(layer)
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(nodes, pos))
        return pos

    msg = "align must be either vertical or horizontal."
    raise ValueError(msg)


pos_0 = multiline_layout(twitter,subset_key='index',scale = 1)


labels0 = {}
for i in range(len(graph[0]['edges'])):
    node1_index = i
    node1_edge = graph[0]['edges'][i]
    for j in range(len(node1_edge)):
        if node1_edge[j] != '':
            node2_index = j
            twitter.add_edge(graph[0]['nodes'][i]['word'], graph[0]['nodes'][j]['word'], color='b')
            labels0[(graph[0]['nodes'][i]['word'], graph[0]['nodes'][j]['word'])] = node1_edge[j]

labels1 = {}
for i in range(len(graph[1]['edges'])):
    node1_index = i
    node1_edge = graph[1]['edges'][i]
    for j in range(len(node1_edge)):
        if node1_edge[j] != '':
            node2_index = j
            twitter.add_edge(graph[1]['nodes'][i]['word'], graph[1]['nodes'][j]['word'], color='b')
            labels1[(graph[1]['nodes'][i]['word'], graph[1]['nodes'][j]['word'])] = node1_edge[j]

word2vec_embeddings = np.load('../data/obama/interim/word2vec_embeddings_obama.npy', allow_pickle=True)
bert_embeddings = np.load('../data/obama/interim/bert_embeddings_obama.npy', allow_pickle=True)


# link nearest node in two graphs

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2)


def get_node_embeddings(graph, embeddings):
    node_features = []
    for node in graph['nodes']:
        node_features.append(embeddings[[node['index']]].mean(axis=0))
    return np.array(node_features)


# embedding0 = get_node_embeddings(graph[0], word2vec_embeddings[0])
# embedding1 = get_node_embeddings(graph[1], word2vec_embeddings[1])

embedding0 = get_node_embeddings(graph[0], bert_embeddings[0])
embedding1 = get_node_embeddings(graph[1], bert_embeddings[1])

w_match = 1
w_nonmatch = 0.5

node_match = []
word_match = []
for i1, e1 in enumerate(embedding1):
    node_sim = []
    for e0 in embedding0:
        if (np.all(e0 == 0)) | (np.all(e1 == 0)):
            node_sim.append(0)
        else:
            node_sim.append(cosine_similarity(e0, e1))
    max_sim_node = np.argmax(node_sim)
    node_match.append((max_sim_node, i1))
    # word_match.append((graph[0][max_sim_node]['word'], graph[1][i1]['word']))

for (i, j) in node_match:
    twitter.add_edge(graph[0]['nodes'][i]['word'], graph[1]['nodes'][j]['word'], color='r')


plt.figure(figsize=(10,10))
edges = twitter.edges()
colors = [twitter[u][v]['color'] for u,v in edges]

nx.draw(twitter, pos_0,connectionstyle='arc3,rad=0.5', edge_color=colors)
nx.draw_networkx_nodes(twitter, pos_0)
nx.draw_networkx_labels(twitter, pos_0)
plt.show()