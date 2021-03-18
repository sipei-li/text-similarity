import random
import time
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler
import torch
import os, json
import numpy as np
import pandas as pd
from nltk import sent_tokenize
from tqdm.autonotebook import trange


from train import train
from args import Args
from model import create_models
from utils import create_dirs
import preprocess
import utils

if __name__ == '__main__':
    args = Args()
    args = args.update_args()

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    model = create_models(args)
    print('model created')

    if args.mode == 'train':
        create_dirs(args)
        # save args
        with open(os.path.join(args.experiment_path, "configuration.txt"), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        # train model
        data_path = f'data/{args.data_name}/processed/{args.phrase_extraction}_{args.negative_selection}{args.negative_number}_kp_from_first_sent_{str(args.phrase_from_first_sent)}_sample_from_same_article_{str(args.sample_from_same_article)}.csv'
        if os.path.exists(data_path):
            print(f'read data from {data_path}')
            df = pd.read_csv(data_path).dropna().reset_index(drop=True)
        else:
            df = pd.read_csv(f'data/{args.data_name}/interim/{args.data_name}.csv')
            df = df.dropna()
            # sample the articles with the most sentences
            df['len_sent'] = df['text'].apply(sent_tokenize).apply(len)
            df = df.sort_values('len_sent', ascending=False).iloc[:20000]
            text_list = df['text'].tolist()
            df = preprocess.generate_negative_samples(args, text_list)
            df = preprocess.convert_into_modeling_df(args, df).dropna().reset_index(drop=True)
            print('negative data generated')
            df.to_csv(data_path, index=False)
            print(f'data saved to {data_path}')
            
        # df = pd.read_csv('data/financial_news/interim/query_1_sentences_random.csv')
        # df['label'] = np.where(df.whole_sim>0, 1, 0)

        sentences1 = df.sentence1.values.tolist()
        sentences2 = df.sentence2.values.tolist()
        labels = df.label
        print('data read completed')

        features1 = {'input_ids':[], 'token_type_ids':[], 'attention_mask':[]}
        features2 = {'input_ids':[], 'token_type_ids':[], 'attention_mask':[]}
        for start_index in trange(0, len(sentences1), args.batch_size, desc="Batches"):
            sentences_batch1 = sentences1[start_index:start_index+args.batch_size]
            feat1 = model.model0.tokenizer(sentences_batch1, max_length=args.max_len,
                                        padding='max_length', truncation=True, return_tensors='pt')
            features1['input_ids'].extend(feat1['input_ids'])
            features1['token_type_ids'].extend(feat1['token_type_ids'])
            features1['attention_mask'].extend(feat1['attention_mask'])

            sentences_batch2 = sentences2[start_index:start_index+args.batch_size]
            feat2 = model.model0.tokenizer(sentences_batch2, max_length=args.max_len,
                                        padding='max_length', truncation=True, return_tensors='pt')
            features2['input_ids'].extend(feat2['input_ids'])
            features2['token_type_ids'].extend(feat2['token_type_ids'])
            features2['attention_mask'].extend(feat2['attention_mask'])
        
        features1['input_ids'] = torch.stack(features1['input_ids'])
        features1['token_type_ids'] = torch.stack(features1['token_type_ids'])
        features1['attention_mask'] = torch.stack(features1['attention_mask'])

        features2['input_ids'] = torch.stack(features2['input_ids'])
        features2['token_type_ids'] = torch.stack(features2['token_type_ids'])
        features2['attention_mask'] = torch.stack(features2['attention_mask'])
        
        labels = torch.tensor(labels)
        print('feature encoding completed')

        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(features1['input_ids'], 
                                features1['attention_mask'],
                                features2['input_ids'],
                                features2['attention_mask'], 
                                labels)

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = args.batch_size # Trains with this batch size.
                )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        eval_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = args.batch_size # Evaluate with this batch size.
                )
        print('dataloader created')

        # total_steps = len(train_dataloader) * args.epochs
        print('start training model')
        train(args, model, train_dataloader, eval_dataloader)
    elif args.mode=='eval':
        print(args.mode)
        fname, epoch = utils.get_last_checkpoint(args, epoch=args.epochs_end)
        utils.load_model(path=fname, device=args.device, model=model)
        print('Model loaded')

    # calcualte similarity
    print('start calculating similarity')
    data_path = f'data/{args.data_name}/processed/eval.csv'
    eval_df = pd.read_csv(data_path)
    sentences1 = (eval_df.sentence1 + ' ' + eval_df.key_phrase).values.tolist()
    sentences2 = (eval_df.sentence2 + ' ' + eval_df.key_phrase).values.tolist()
    labels = eval_df.sim.values

    embedding1 = model.encode(sentences1)
    embedding2 = model.encode(sentences2)
    print( 'sentence embedding finished')
    similarity = list(map(utils.cosine_similarity, embedding1, embedding2))
    print('simiarity finished')
    corr = np.corrcoef(labels, np.array(similarity))
    corr_df = pd.DataFrame(columns=['correlation'])
    corr_df.correlation = corr[0,1]
    corr_df.to_csv(args.logging_corr_path, index=False)
    print(corr)[0,1]


    