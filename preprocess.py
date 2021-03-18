import nltk
import glob
import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('msmarco-distilroberta-base-v2')


def sample_rand_consec3_max_like_kp(article_rows):

    for row in article_rows.itertuples():
        high = [0,row]
        high2 = [0,row]

        #get random key phrase
        curr_words = nltk.word_tokenize(row[1])

        if(len(curr_words) - 3 == 0):
            random_kp_i = 0
        elif(len(curr_words) - 3 < 0):
            continue
        else:
            random_kp_i = random.randint(0, len(curr_words) - 3)

        attempts = 0

        #make sure the key phrase is made up of words, not punctuation
        while(not curr_words[random_kp_i].isalpha() or not curr_words[random_kp_i + 1].isalpha() or 
                                                              not curr_words[random_kp_i + 2].isalpha()):
            random_kp_i = random.randint(0, len(curr_words) - 3)

            if attempts > 10:
                break
            attempts += 1

        #in case we were not able to successfully sample a key phrase
        if(not curr_words[random_kp_i].isalpha() or not curr_words[random_kp_i + 1].isalpha() or 
                                                              not curr_words[random_kp_i + 2].isalpha()):
            continue


    #     print(curr_words[random_kp_i] + " " + curr_words[random_kp_i + 1] + " " + curr_words[random_kp_i + 2])
        row_embed = model.encode(curr_words[random_kp_i] + " " + curr_words[random_kp_i + 1] + " " + curr_words[random_kp_i + 2])

        #now sample random sentences
        sample = article_rows.sample(n=10, replace=True)

        for sample_row in sample.itertuples():
            if row[2] != sample_row[1]:

                sample_embed = model.encode(sample_row[1])
                curr_sim = [util.pytorch_cos_sim(row_embed, sample_embed), sample_row[1]]
                
                if(curr_sim[1] == high[1] or curr_sim[1] == high2[1]):
                    continue
                
                if (curr_sim[0] > high[0]):
                    high2 = high
                    high = curr_sim
                elif (curr_sim[0] > high2[0]):
                    high2 = curr_sim
                    
#         print("SAMPLING COMPLETE: ")
#         print(high)
#         print(high2)

        article_rows.at[row[0], 'key_phrase'] = curr_words[random_kp_i] + " " + curr_words[random_kp_i + 1] + " " + curr_words[random_kp_i + 2]
        article_rows.at[row[0], 'n1'] = high[1]
        article_rows.at[row[0], 'n2'] = high2[1]
#         print("KEY PHRASE: ", article_rows.at[row[0], 'key_phrase'])
        
    return article_rows


def generate_sample_from_same_article(text_list, sample_func):
    corpus = []
    length = len(text_list)
    i = 1
    for row in text_list:
        rows_list = []
        sentences = nltk.sent_tokenize(row)
        prev = '\n' # trick to make it so we skip the first element
        for curr in sentences:
            if '\n' not in prev:
                if '\n' not in curr:
                    dict1 = {'s1': prev, 's2': curr, 'key_phrase': "", 'n1': "", 'n2': ""}
                    rows_list.append(dict1)
            prev = curr
        updated_sample = sample_func(pd.DataFrame(rows_list, columns=['s1', 's2','key_phrase','n1','n2']))
        
        corpus.append(updated_sample)
        print(i, "/", length)
        i += 1
    result_df = pd.concat(corpus)
    return result_df


def generate_negative_samples(args, text_list):
    if (args.phrase_extraction=='rand_consec3') and (args.negative_selection=='max_like_phrase'):
        if args.sample_from_same_article:
            df = generate_sample_from_same_article(text_list, sample_rand_consec3_max_like_kp)
            return df


def convert_into_modeling_df(args, df):
    if args.phrase_from_first_sent:
        df['sentence1'] = df[['s1', 'key_phrase']].agg(' '.join, axis=1)
        df = df.rename(columns={'s2':'sentence2'})

        model_df1 = df[['sentence1', 'n1']]
        model_df1 = model_df1.rename(columns={'n1':'sentence2'})
        model_df2 = df[['sentence1', 'n2']]
        model_df2 = model_df2.rename(columns={'n2':'sentence2'})
        model_df = pd.concat([model_df1, model_df2], sort=False)
        model_df['label'] = 0

    else:
        df['sentence2'] = df[['s2', 'key_phrase']].agg(' '.join, axis=1)
        df = df.rename(columns={'s1':'sentence1'})

        model_df1 = df[['n1', 'sentence2']]
        model_df1 = model_df1.rename(columns={'n1':'sentence1'})
        model_df2 = df[['n2', 'sentence2']]
        model_df2 = model_df2.rename(columns={'n2':'sentence1'})
        model_df = pd.concat([model_df1, model_df2], sort=False)
        model_df['label'] = 0

    model_df3 = df[['sentence1', 'sentence2']]
    model_df3['label'] = 1

    model_df = pd.concat([model_df, model_df3], sort=False)
    return model_df

    