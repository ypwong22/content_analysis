# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:10:45 2018

@author: wangy

https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
"""
##import re
import nltk
from nltk import word_tokenize
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora
import os
import numpy as np
import matplotlib.pyplot as plt


# Get the stem of a word. 
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(word)


# Split text into tokens (words & punctuations).
def tokenize_clean(text, excluded_words):
    en_stop = set(nltk.corpus.stopwords.words('english'))
    en_stop = en_stop.union(set(excluded_words))

    tokens = word_tokenize(text)

    # ---- remove verb
    tagged = nltk.pos_tag(tokens)
    ## ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'CC', 'IN', 'RB', 'RBR', 
    ##  'RBS']
    tokens = [t[0] for t in tagged if t[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'FW']]

    # ---- remove stop words and a list of excluded words
    tokens = [get_lemma(t.lower()) for t in tokens]
    cleaned_text = [t for t in tokens if t not in en_stop and t.isalpha()]


    # ---- stem-ize in order to harmonize differences between noun, adj.,
    #      adv. etc forms
    stemmed_text = np.array([get_lemma2(token) for token in cleaned_text])

    # ---- identify the unique stemmed_text and re-map into the 
    #      original cleaned text, to improve readability
    stemmed_text2, stem_unique_index = np.unique(stemmed_text, \
                                                 return_index = True)
    
    cleaned_text = np.array(cleaned_text)
    cleaned_text2 = cleaned_text[stem_unique_index]

    return cleaned_text2


# Count the number of lines in a file.
def file_len(fname):
    i = -1
    f = open(fname, 'r', encoding='utf-8')
    for i, l in enumerate(f):
        pass
    f.close()
    return i + 1


# File path setups. ###########################################################
path_input = 'P:/ene.yssp/Yaoping_Wang/Applications/2017-18 Job Applications/2018-10-08 Microsoft/data'
path_temp = 'P:/ene.yssp/Yaoping_Wang/Applications/2017-18 Job Applications/2018-10-08 Microsoft/intermediate'
path_out = 'P:/ene.yssp/Yaoping_Wang/Applications/2017-18 Job Applications/2018-10-08 Microsoft/output'

##query = 'climate_resilience'
query = 'infrastructure_resilience'
path_temp = os.path.join(path_temp, query)


excluded_words = np.genfromtxt(os.path.join(path_input, 'excluded_words.txt'), dtype=str)


# List of journal x year records. #############################################
records = os.listdir(path_temp)


# Tokenize the text by journal and year list. #################################
jour_list = []
year_list = []
for rr in records:
    jour_list.append(rr.split('_')[0])
    year_list.append(rr.split('_')[1].split('.')[0])
jour_list = np.array(jour_list)
year_list = np.array(year_list)

year_list_unique = np.unique(year_list)
n_years = len(year_list_unique)

n_studies_year = [0] * n_years
for rr in records:
    year_temp = rr.split('_')[1].split('.')[0]
    place_temp = np.where([x==year_temp for x in year_list_unique])[0][0]
    n_studies_year[place_temp] += file_len(os.path.join(path_temp, rr))


tokenized_data = []
for rr in records:
    f = open(os.path.join(path_temp, rr), 'r', encoding = 'utf-8')
    text = f.read()
    f.close()

    tokens = tokenize_clean(text, excluded_words)
    tokenized_data.append(tokens)


# Word frequency: Overall. ####################################################
tokenized_data_jumble = [y for x in tokenized_data for y in x]
cf = nltk.FreqDist(tokenized_data_jumble)
f = open(os.path.join(path_out, 'word_frequency_all.csv',), 'w')
for word, frequency in cf.most_common(1000):
    f.write(u'{}, {}\n'.format(word, frequency))
f.close()


# Topic modeling: Overall. ####################################################
n_topics = 10
# ---- build a Dictionary - association word to numeric id
common_dict = corpora.Dictionary(tokenized_data)
# ---- transform the collection of texts to a numerical form
corpus = [common_dict.doc2bow(text) for text in tokenized_data]
## ---- have a look at how the 20th document looks like: [(word_id, count), ...]
## print(corpus[20])
## [(12, 3), (14, 1), (21, 1), (25, 5), (30, 2), (31, 5), (33, 1), (42, 1), (43, 2),  ...
# ---- build the LDA model
lda_model = models.LdaModel(corpus=corpus, num_topics=n_topics, id2word=common_dict)
lda_model.show_topics()
# ---- build the LSI model
lsi_model = models.LsiModel(corpus=corpus, num_topics=n_topics, id2word=common_dict)
lsi_model.show_topics()


# Topic modeling: Temporal trends. ############################################
tokenized_data_year = []
for yy in year_list_unique:
    temp = []
    for i in range(len(tokenized_data)):
        if (year_list[i] == yy):
            temp.extend(tokenized_data[i])
    tokenized_data_year.append(temp)    
# ---- transform the collection of texts to a numerical form
corpus_year = [common_dict.doc2bow(text) for text in tokenized_data_year]
# ---- get topic probability distribution for a document
topic_year = np.empty([lda_model.num_topics, n_years])
for yy in range(n_years):
    vector = lda_model[corpus_year[yy]]
    for vv in vector:
        topic_year[vv[0], yy] = vv[1]
# ---- display the temporal trend
fig, ax = plt.subplots(1, 2, figsize=(8,3.5))
p0 = ax[0].plot(year_list_unique, n_studies_year)
ax[0].set_title('No. Studies per Year')
pos = ax[1].imshow(topic_year, cmap = 'RdYlBu', vmin=0, vmax=1, \
                    aspect = 0.8)
ax[1].set_xticks(range(n_years))
ax[1].set_xticklabels(year_list_unique)
ax[1].set_yticks(range(n_topics))
ax[1].set_title('Topics over Time')
cax = fig.add_axes([0.93, 0.1, 0.03, 0.8])
fig.colorbar(pos, cax=cax, orientation = 'vertical')
