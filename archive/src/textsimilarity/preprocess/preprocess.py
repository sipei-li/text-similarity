import pandas as pd 
import os 
import sys 

DATA_PATH = sys.argv[1] 
OUT_DATA_NAME = sys.argv[2]
RAW_DIR = os.path.dirname(DATA_PATH)

if "imdb" in DATA_PATH:
    data = pd.read_csv(DATA_PATH)
    
    data = data.rename(columns={"Column1":"label", "Column2":"sentence"})

    data.to_csv(os.path.join(RAW_DIR, f'{OUT_DATA_NAME}.csv'), index=False)
else:
    data = pd.read_csv(DATA_PATH, sep=r'\t', header=None, engine='python')

    data = data.rename(columns={0:"label", 1:"sentence"})
    
    # data_shuffled = data.sample(frac=1, random_state=1).reset_index(drop=True)

    # train_len = round(0.7*len(data_shuffled))
    # test_len = round((len(data_shuffled)-train_len)/2)

    # data_train = data_shuffled.iloc[:train_len]
    # data_test = data_shuffled.iloc[train_len:train_len+test_len]
    # data_val = data_shuffled.iloc[train_len+test_len:]

    # data_train.to_csv(os.path.join(RAW_DIR, 'train.csv'), index=False)
    # data_test.to_csv(os.path.join(RAW_DIR, 'test.csv'), index=False)
    # data_val.to_csv(os.path.join(RAW_DIR, 'dev.csv'), index=False)
    data.to_csv(os.path.join(RAW_DIR, f'{OUT_DATA_NAME}.csv'), index=False)