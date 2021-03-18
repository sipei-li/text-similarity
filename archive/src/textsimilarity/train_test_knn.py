import pandas as pd
import numpy as np
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    DATA_DIR = sys.argv[1]
    data_name = sys.argv[2]

    train = pd.read_csv(os.path.join('data', DATA_DIR, 'raw', 'train.csv'))
    test = pd.read_csv(os.path.join('data', DATA_DIR, 'raw', 'test.csv'))

    if data_name == 'wmd':
        training_distance = np.load(os.path.join('data', DATA_DIR, 'processed', f'{data_name}_train.npy'))
        testing_distance = np.load(os.path.join('data', DATA_DIR, 'processed', f'{data_name}_test.npy'))
    elif data_name == 'g_dist':
        threshold = sys.argv[3]
        training_distance = np.load(os.path.join('data', DATA_DIR, 'processed', f'{data_name}_train_word2vec_threshold{threshold}.npy'))
        testing_distance = np.load(os.path.join('data', DATA_DIR, 'processed', f'{data_name}_test_word2vec_threshold{threshold}.npy'))

    knn = KNeighborsClassifier(metric="precomputed", n_neighbors=10)
    knn.fit(training_distance, train.label)

    # pred_test = knn.predict(testing_distance)
    print(knn.score(testing_distance, test.label))