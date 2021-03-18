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
nltk.download('stopwords')

from preprocess.get_coref_and_dp import get_neural_coreference
from get_bert_features import pad_to_max_length
import spacy
from spacy import displacy
nlp = spacy.load("en")
import neuralcoref 
neuralcoref.add_to_pipe(nlp)
from matplotlib import pyplot as plt
from get_similarity import cosine_similarity

stop_words = set(stopwords.words('english')) 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

vocab = tokenizer.get_vocab()
vocab_list = list(vocab.keys())

"""
visualize dependency tree and nearest words in two sentences.
usage: import into jupyter notebook.
"""

def word2vec_visualize(doc1, doc2):
    
    text1 = doc1
    text2 = doc2 

    coref1 = get_neural_coreference(doc1)
    coref2 = get_neural_coreference(doc2)
    
    doc1 = [j for n in coref1 for j in n.split()]
    doc2 = [j for n in coref2 for j in n.split()]

    word_list1 = doc1
    word_list2 = doc2

    # get word2vec embedding
    emb1 = [nlp(w).vector for w in doc1]
    emb2 = [nlp(w).vector for w in doc2]
    
    # result1 = [(word, emb) for word, emb in zip(doc1,emb1) if (word not in stop_words) and (word in vocab_list)]
    result1 = [(word, emb) for word, emb in zip(doc1,emb1) if (word not in stop_words)]
    doc1 = [x[0] for x in result1]
    emb1 = [x[1] for x in result1]
    
    # result2 = [(word, emb) for word, emb in zip(doc2,emb2) if (word not in stop_words) and (word in vocab_list)]
    result2 = [(word, emb) for word, emb in zip(doc2,emb2) if (word not in stop_words)]
    doc2 = [x[0] for x in result2]
    emb2 = [x[1] for x in result2]
    
    best_word = {}
    for i2, e2 in enumerate(emb2):
        word_sim = []
        for e1 in emb1:
            word_sim.append(cosine_similarity(e1, e2))
        i1 = np.argmax(word_sim)
        best_word[doc2[i2]] = doc1[i1]
    
    text1 = nlp(text1)
    text2 = nlp(text2)
    displacy.render(text1, style="dep", options={'distance':100, 'compact':True}, jupyter=True)
    displacy.render(text2, style="dep", options={'distance':100, 'compact':True}, jupyter=True)

    # define drawing of the words and links separately.
    def plot_words(wordlist, col, ax):
        bbox_props = dict(boxstyle="round4,pad=0.3", fc="none", ec="b", lw=2)
        # wordlist = wordlist.reverse()
        for i, word in enumerate(wordlist):
            ax.text(col, i, word, ha="center", va="center",
                    size=12, bbox=bbox_props)

    def plot_links(list1, list2, cols, ax):
        connectionstyle = "arc3,rad=0"
        for i, word in enumerate(list1):
            if word in best_word.keys():
                word2 = best_word[word]
                try:
                    j = list2.index(word2)
                except ValueError:
                    continue
                # define coordinates (relabelling here for clarity only)
                y1, y2 = i, j
                x1, x2 = cols
                # draw a line from word in 1st list to word in 2nd list
                ax.annotate("", xy=(x2, y2), xycoords='data',
                            xytext=(x1, y1), textcoords='data',
                            arrowprops=dict(
                                arrowstyle="->", color="k", lw=2,
                                shrinkA=25, shrinkB=25, patchA=None, patchB=None,
                                connectionstyle=connectionstyle,))
            else:
                continue

    # now plot them all -- words first then links between them
    plt.figure(figsize=(10,10))
    plt.figure(1); plt.clf()
    fig, ax = plt.subplots(num=1)

    plot_words(word_list2, col=0, ax=ax)
    plot_words(word_list1, col=1, ax=ax)
    # plot_words(list3, col=0, ax=ax)
    plot_links(word_list2, word_list1, ax=ax, cols=[0,1])
    # plot_links(list1, list3, ax=ax, cols=[1,0])

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, max(len(word_list2),len(word_list1))+0.5)
    plt.show()

    print(best_word)

def bert_visualize(doc1, doc2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    text1 = doc1
    text2 = doc2 

    coref1 = get_neural_coreference(doc1)
    coref2 = get_neural_coreference(doc2)
    
    doc1 = [j for n in coref1 for j in n.split()]
    doc2 = [j for n in coref2 for j in n.split()]

    word_list1 = doc1
    word_list2 = doc2

    input_token_ids1 = torch.tensor(tokenizer.convert_tokens_to_ids([j if j in vocab_list else tokenizer.tokenize(j)[0] for j in doc1]))
    input_token_ids2 = torch.tensor(tokenizer.convert_tokens_to_ids([j if j in vocab_list else tokenizer.tokenize(j)[0] for j in doc2]))
    
    input_token_ids = [input_token_ids1,input_token_ids2]

    input_token_ids = pad_to_max_length(input_token_ids, 128)
    dataset = TensorDataset(input_token_ids)
    dataloader = DataLoader(
                dataset,
                sampler = SequentialSampler(dataset),
                batch_size = 1 
            )

    bert_embeddings = []
    with torch.no_grad():
        for b in dataloader:
            bert_embeddings.extend(bert_model(b[0].type(torch.LongTensor).to(device))[0].cpu().numpy())

    bert_embeddings = np.array(bert_embeddings) # (2, 128, 768)
    emb1 = bert_embeddings[0]
    emb2 = bert_embeddings[1]
    
    # result1 = [(word, emb) for word, emb in zip(doc1,emb1) if (word not in stop_words) and (word in vocab_list)]
    result1 = [(word, emb) for word, emb in zip(doc1,emb1) if (word not in stop_words)]
    doc1 = [x[0] for x in result1]
    emb1 = [x[1] for x in result1]
    
    # result2 = [(word, emb) for word, emb in zip(doc2,emb2) if (word not in stop_words) and (word in vocab_list)]
    result2 = [(word, emb) for word, emb in zip(doc2,emb2) if (word not in stop_words)]
    doc2 = [x[0] for x in result2]
    emb2 = [x[1] for x in result2]
    
    best_word = {}
    for i2, e2 in enumerate(emb2):
        word_sim = []
        for e1 in emb1:
            word_sim.append(cosine_similarity(e1, e2))
        i1 = np.argmax(word_sim)
        best_word[doc2[i2]] = doc1[i1]
    
    text1 = nlp(text1)
    text2 = nlp(text2)
    displacy.render(text1, style="dep", options={'distance':100, 'compact':True}, jupyter=True)
    displacy.render(text2, style="dep", options={'distance':100, 'compact':True}, jupyter=True)

    # define drawing of the words and links separately.
    def plot_words(wordlist, col, ax):
        bbox_props = dict(boxstyle="round4,pad=0.3", fc="none", ec="b", lw=2)
        # wordlist = wordlist.reverse()
        for i, word in enumerate(wordlist):
            ax.text(col, i, word, ha="center", va="center",
                    size=12, bbox=bbox_props)

    def plot_links(list1, list2, cols, ax):
        connectionstyle = "arc3,rad=0"
        for i, word in enumerate(list1):
            if word in best_word.keys():
                word2 = best_word[word]
                try:
                    j = list2.index(word2)
                except ValueError:
                    continue
                # define coordinates (relabelling here for clarity only)
                y1, y2 = i, j
                x1, x2 = cols
                # draw a line from word in 1st list to word in 2nd list
                ax.annotate("", xy=(x2, y2), xycoords='data',
                            xytext=(x1, y1), textcoords='data',
                            arrowprops=dict(
                                arrowstyle="->", color="k", lw=2,
                                shrinkA=25, shrinkB=25, patchA=None, patchB=None,
                                connectionstyle=connectionstyle,))
            else:
                continue

    # now plot them all -- words first then links between them
    plt.figure(figsize=(10,10))
    plt.figure(1); plt.clf()
    fig, ax = plt.subplots(num=1)

    plot_words(word_list2, col=0, ax=ax)
    plot_words(word_list1, col=1, ax=ax)
    # plot_words(list3, col=0, ax=ax)
    plot_links(word_list2, word_list1, ax=ax, cols=[0,1])
    # plot_links(list1, list3, ax=ax, cols=[1,0])

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, max(len(word_list2),len(word_list1))+0.5)
    plt.show()

    print(best_word)