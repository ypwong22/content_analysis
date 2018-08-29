# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 09:56:38 2018

@author: wangy

Environment: python 3.6

Pull articles (title, journal, year, key words, abstracts) from Scopus

Key words: climate resilience
"""
from pyscopus import Scopus
import json
import os
from datetime import datetime
import numpy as np
import itertools as it
import re


# Search key words (limit to article, review, or in-press)
##search_term = 'TITLE-ABS-KEY ( climate AND resilience )  AND  ' + \
##              '( DOCTYPE(ar)  OR  DOCTYPE(re) OR DOCTYPE(ip) )'
search_term = 'TITLE-ABS-KEY ( infrastructure AND resilience )  AND  ' + \
              '( DOCTYPE(ar)  OR  DOCTYPE(re) OR DOCTYPE(ip) )'

# File paths
path_root = 'P:/ene.yssp/Yaoping_Wang/Applications/2017-18 Job Applications/2018-10-08 Microsoft/'
path_json = os.path.join(path_root, 'data')
path_out = os.path.join(path_root, 'intermediate', 'infrastructure_resilience')


# Load API key configuration
con_file = open(os.path.join(path_json, "config.json"))
config = json.load(con_file)
con_file.close()


# Conduct search
scopus = Scopus(config['apikey'])
search_df = scopus.search(search_term, count=5000)


# Save the search results by journal & year of publication
# ---- only save journals that have at least 2 papers
jour_list = []
for xx in search_df['publication_name'].unique():
    if (sum(search_df['publication_name'] == xx) > 1):
        # ---- adjust some irregularities in journal names
        xx2 = re.sub('[^a-zA-Z]+', ' ', xx)
        xx2 = xx2.strip(' ')
        if ((len(path_out) + len(xx2)) > 200):
            xx2 = xx2[:(200 - len(path_out))]
        search_df.loc[search_df['publication_name'] == xx, 'publication_name'] = xx2
        jour_list.append(xx2)


year_list = []
for xx in search_df['cover_date']:
    pub_date = datetime.strptime(xx, '%Y-%m-%d')
    year_list.append(pub_date.year)

year_list = np.array(year_list)
# ---- simple frequency of the years of publication
y = np.bincount(year_list)
ii = np.nonzero(y)[0]
print(np.vstack((ii,y[ii])).T)


##
##jour_list = jour_list[261:]
##jour_list[0] = re.sub('[^a-zA-Z]+', ' ', jour_list[0])
##jour_list[0] = jour_list[0].strip(' ')
##if ((len(path_out) + len(jour_list[0])) > 200):
##    jour_list[0] = jour_list[0][:(200 - len(path_out))]
##search_df.loc[search_df['publication_name'] == jour_list[0], 'publication_name'] = jour_list[0]

for xx,yy in it.product(jour_list, np.unique(year_list)):
    temp = (search_df['publication_name'] == xx) & \
           (year_list == yy)
    if (sum(temp) > 0):
        search_df_sub = search_df.loc[temp, :].reset_index()

        f = open(os.path.join(path_out, xx + '_' + str(yy) + '.txt'), 'w', \
                 encoding='utf-8')

        for index, zz in search_df_sub.iterrows():
            try:
                pub_info = scopus.retrieve_abstract(zz['scopus_id'])
            except:
                # If cannot retrieve abstract then skip.
                continue

            ##pub_jour = pub_info['prism:publicationName']
            ##pub_date = datetime.strptime(pub_info['prism:coverDate'], '%Y-%m-%d')
            pub_text = pub_info['title'] + pub_info['abstract']

            pub_text = pub_text.replace('\n', ' ').replace('©', ' ')

            f.write(pub_text + '\n')

        f.close()
