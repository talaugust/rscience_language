
# coding: utf-8

# In[2]:


import os
import sys
import numpy as np
import csv as csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import json
from datetime import datetime
from termcolor import colored
from sklearn.model_selection import train_test_split
from langdetect import detect
import random
import seaborn as sns


from nltk.util import bigrams, ngrams
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.lm.smoothing import GoodTuring
from nltk.tokenize import word_tokenize
from nltk.lm.models import InterpolatedLanguageModel, WittenBellInterpolated, KneserNeyInterpolated, MLE, Laplace
from nltk.lm.api import LanguageModel
from nltk.lm import NgramCounter

# TODO: Take this out
# try:
#     os.chdir('../data/test_subs_2017/')
#     print('changed directory to ../data/test_subs_2017/')
# except OSError:
#     print('directory already ../data/test_subs_2017/')

MONTHS = list(range(1,13))

# # SLM building functions
# How different (in terms of cross-entrop) are posts from users within and outside a community?
# Based on no country and Justine's later paper, right now we will look at, within a certian month, 
# 
# Follow this: 
#     
# - randomly sampling 200 active users—defined as users with at least 5 comments in the respective community and month.
# 
# - For each of these 200 active users we select 5 10-word spans from 5 unique comments.
# 
# - To ensure robustness and maximize data efficiency, we construct 100 SLMs for each community-month pair that
# has enough data, bootstrap-resampling (resampling with replacement) from the set of active users.

# In[3]:


##################################################################
# randomly sampling 200 users  
# defined as users with at least 5 in the respective community and month.
# Kind defines if you are looking at comments or posts, if None then using both

# TODO: might not want to reset the author_df, maybe make a copy?
##################################################################
def get_active_users(author_df, month, author_col, threshold=5, num_authors=200, kind=None):
    if kind:
        author_df = author_df[author_df['kind'] == kind]
    if num_authors: 
        print(len(author_df[author_df[month] > threshold]))
        return author_df[author_df[month] > threshold].drop_duplicates().sample(num_authors)[author_col]
    else: 
        return author_df[author_df[month] > threshold].drop_duplicates()[author_col]
    
    
##################################################################   
# For each of these 200 active users
# select 5 random 10-word spans from 5 unique comments
# ASSUMING: from that month
# ASSUMING: this is one 10 word span from each comment, not 5 for each comment
##################################################################
def get_random_span(text, length):
    text = [w for w in word_tokenize(text)]
    try: 
        beg = random.randint(0, len(text) - length)
        end = beg + 10
        return text[beg:end]
    except:
        raise IndexError("Error: index out of range, probably happened if you didn't clean comments to be at least 10 words long")

# def get_train_active_user_comments(df, authors, month, month_col='created_month'):
#     df_month = df[df[month_col] == month]
#     df_month_author = df_month[df_month['author'].apply(lambda x: x in authors)]
#     df_grouped = df_month_author.groupby('author')
#     sampled_comments = []
#     for a, g in df_grouped:
#         sample = g.sample(5)['body'].apply(lambda x: word_tokenize(x)[:10]) # TODO: change to random span
#         sampled_comments.extend(sample)
#     return sampled_comments

##################################################################
# To ensure robustness and maximize data efficiency, 
# we construct 100 SLMs for each community-month pair that has enough data
##################################################################
def construct_LM(text, vocab=None):
    if vocab:
        train, _ =  padded_everygram_pipeline(2, text)
    else: 
        train, vocab =  padded_everygram_pipeline(2, text)
    lm = Laplace(2)
    lm.fit(train, vocab)
    return lm


# build an SLMs for a single month
def build_SLMs(df, author_counts, slm_count, month):
    print('Creating ', colored(str(slm_count) + ' SLMs ', 'red'), 'for', colored(' month ' + str(month), 'green'), '.....', end=' ')
    slms = []
    for i in range(1, slm_count + 1):
        active_users = get_active_users(author_counts, str(month), 'author')
        active_user_comments = get_user_comments(df, list(active_users), month=month, num_posts=5)
        slm = construct_LM(active_user_comments)
        slms.append(slm)
    print('Done')
    return slms

# returns dict of {month:SLM}
def build_monthly_SLM(df, author_counts, slm_count,):
    slm_dict = {}
    for m in MONTHS:
        slms = build_SLMs(df, author_counts, slm_count, month=m)
        slm_dict[m] = slms
    return slm_dict




# # Acculturation Gap functions
# 
# We compute a basic measure of the acculturation gap for a community-month
# relative difference of the cross-entropy of comments by users active in
# singleton comments by are still activein Reddit in general
# 
# For each bootstrap-sampled SLM we compute the cross-entropy of
# 50 comments by active users (10 comments from 5 randomly sampled active users, who
# were not used to construct the SLM) and 50 comments from randomly-sampled outsiders.
# 
# Acc gap for a specific month is exp_value of outsiders - exp_value of active users / exp_value of active users 

# In[4]:


##################################################################
# getting outsiders -- users who only ever posted once in a community-- 
# but are still activein Reddit in general  (TODO)
##################################################################
def get_outside_users(author_df, month, author_col, threshold=1, num_authors=None, kind=None):
    if kind:
        author_df = author_df[author_df['kind'] == kind]
    outside_users = author_df[author_df[[str(m) for m in MONTHS]].sum(axis=1) <= threshold]
    if num_authors:
        return outside_users[outside_users[month] == threshold][author_col].sample(num_authors)
    else: 
        return outside_users[outside_users[month] == threshold][author_col]

##################################################################
# get 50 comments by active users (10 comments from 5 randomly sampled active users, 
# who were not used to construct the SLM -TODO) and 50 comments from randomly-sampled outsiders
# same length controlling effects used here - select random 10 word span from each comment (see ASSUMING above)
##################################################################
def get_user_comments(df, authors, month, num_posts, month_col='created_month'):
    df_month = df[df[month_col] == month]
    df_month_author = df_month[df_month['author'].apply(lambda x: x in authors)]
    df_grouped = df_month_author.groupby('author')
    sampled_comments = []
    for a, g in df_grouped:
        if num_posts:
            sample = g.sample(num_posts)['body'].apply(lambda x: [w for w in word_tokenize(x)][:10]) 
        else: 
            sample = g['body'].apply(lambda x: [w for w in word_tokenize(x)][:10]) 
        sampled_comments.extend(sample)
    return sampled_comments

###################################################################
# Acc gap for a specific month is 
# exp_value of outsiders - exp_value of active users / exp_value of active users
# ASSUMING: Expected value in this case is just a mean, since each likelihood is equally possible
# ISSUE: some spans are < 10
##################################################################

# wrapper function for possibly doing something else than just calculcating entropy
def check_ent(slm, text):
    ent = slm.entropy(text)
    return ent

def calc_single_acc_gap(slms, text):
    entropies = []
    text_bigrams = [list(ngrams(sent, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')) for sent in text]
    for slm in slms:
        slm_entropies = list(map(lambda x: check_ent(slm, x), text_bigrams))
        entropies.append(slm_entropies)
    return entropies

def calc_month_acc_gap(slms_month_dict, month, text):
    slms = slms_month_dict[month]
    return calc_single_acc_gap(slms, text)
  
def calc_acc_gap(slms, author_counts, comments):
    monthly_acc_gap = {}
    for month in slms.keys():
        print('Calculating cross entropy for', colored('month ' + str(month), 'green'), '.....')
        active_authors = get_active_users(author_counts, str(month), 'author', threshold=10, num_authors=5)
        outside_authors = get_outside_users(author_counts, str(month), 'author', threshold=1, num_authors=50)
        print('sampled active users:', len(active_authors), 'sampled outside users:', len(outside_authors))

        active_comments = get_user_comments(comments, list(active_authors), month=month, num_posts=10)
        outside_comments = get_user_comments(comments, list(outside_authors), month=month, num_posts=1)
        print('sampled active comments:', len(active_comments), 'sampled outside comments:', len(outside_comments))

        active_ent = calc_month_acc_gap(slms, month=month, text=active_comments)
        outside_ent = calc_month_acc_gap(slms, month=month, text=outside_comments)
        
        exp_val_active_ent = np.mean(list(flatten(active_ent)))
        exp_val_outside_ent = np.mean(list(flatten(outside_ent)))

        monthly_acc_gap[month] = (exp_val_outside_ent - exp_val_active_ent) / exp_val_active_ent
        print('Saving acc gap for', colored('month ' + str(month), 'green'))
    return monthly_acc_gap
       


# # Build SLMs and calculate Acculturation Gap
# Using all the functions above to actually calculate and plot the acculturaion gaps
# Import different clened csvs of data representing comments of different subs

# In[5]:


###################################################################
# Importing data
###################################################################
def import_csvs(sub, path='data/cleaned/train/2017/', ext='_train_2017.csv', comment_pre_path='data/cleaned/sub_comments/', comment_ext='_comments_2017.csv'):
    
    # currently importing the same comments file for test/train
    # This is because the authors have been seperated, so there shouldn't be any of the same messages
    # between the two sets (even though they pull text from the same file)
    comment_path = comment_pre_path+sub+comment_ext
    
    author_path = path+'author_counts/'+sub+'_author_counts'+ext
    
    print('Importing ', colored(comment_path, 'magenta'),'.....', end=' ')
    df_sub_comments = pd.read_csv(comment_path)
    print('Done')
    print('Importing ', colored(author_path, 'magenta'),'.....', end=' ')
    df_author_counts = pd.read_csv(author_path)
    print('Done')
    
    # renaming month columns per the issue of having a string float
    
    cols = df_author_counts.columns.tolist()
    df_author_counts = df_author_counts.rename(index=str, columns={c:str(int(float(c))) for c in cols[2:len(cols)-1]})
    
    
    return df_sub_comments, df_author_counts


# In[17]:

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage',sys.argv[0], '[num of SLMs per month]')

    slm_num = sys.argv[1]

    subs = ['Naruto', 'pics', 'BabyBumps', 'science', 'politics', 'Cooking']
    print('Running ', slm_num, 'monthly SLMs for', colored(subs, 'magenta'))

    fig, axs = plt.subplots(1, len(subs), sharey=True, tight_layout=True)
    fig.set_figheight(10)
    fig.set_figwidth(50)
    subs_acc = {}
    for i, s in enumerate(subs):
        print('-----------------------------------------------')
        print('Acculturation Gap for ', colored(s, 'magenta'))
        print('-----------------------------------------------')
        
        # TODO: has not been tested
        df_comments, df_author_counts_train = import_csvs(s)
        _, df_author_counts_test = import_csvs(s, path='cleaned/test/2017/', ext='_test_2017.csv')

        slms = build_monthly_SLM(df_comments, df_author_counts_train, 10)

        monthly_acc_gaps = calc_acc_gap(slms, df_author_counts_test, df_comments)

        # Old code that should work
#         df_comments, df_author_counts = import_csvs(s)
#         slms = build_monthly_SLM(df_comments, df_author_counts, int(slm_num))
#         monthly_acc_gaps = calc_acc_gap(slms, df_author_counts, df_comments)

        subs_acc[s] = monthly_acc_gaps

        sns.lineplot(x=list(monthly_acc_gaps.keys()), y=list(monthly_acc_gaps.values()), ax=axs[i])
        axs[i].set_title('Acculturation Gap for ' + s, fontsize=30)
        axs[i].set_xlabel('Month', fontsize=30)
        axs[i].set_ylabel('Gap (Higher = more different)')
        axs[i].tick_params(axis='both', which='major', labelsize=25)

    fig.show()
    fig.savefig('../fig/sub_acc_gaps_monthly')

    print({sub: (np.mean(list(subs_acc[sub].values())), np.std(list(subs_acc[sub].values())))  for sub in subs_acc})
    print(subs_acc)

