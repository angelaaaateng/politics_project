import streamlit as st

import snscrape.modules.twitter as sntwitter
import pandas as pd
import tabula
from tqdm import tqdm
# import twitterscraper
import textstat

#Base and Cleaning 
import json
import requests
import numpy as np
import emoji
import regex
import re
import string
from collections import Counter

#Visualizations
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt 

# import pyLDAvis.gensim
import pyLDAvis
import pyLDAvis.gensim_models
import chart_studio
import chart_studio.plotly as py 
import chart_studio.tools as tls

#Natural Language Processing (NLP)
import spacy
import gensim
from spacy.tokenizer import Tokenizer
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS as SW
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)

from wordcloud import WordCloud

import time


nlp = spacy.load('en_core_web_lg')


st.title('Tweet Scraper')

# read tweets from before 
senator_tweet_df = pd.read_csv('C:/Users/Admin/Documents/GitHub/politics_project/senator_tweets_df.csv')

# st.dataframe(data=senator_tweet_df)

st.write(senator_tweet_df.shape)

print("Running reading scores... ")

my_bar = st.progress(0)
senator_tweet_df['flesch'] = senator_tweet_df['tweet_content'].apply(lambda s:textstat.flesch_reading_ease(s))
my_bar.progress(1/6)
senator_tweet_df['flesch_grade'] = senator_tweet_df['tweet_content'].apply(lambda s:textstat.flesch_kincaid_grade(s))
my_bar.progress(2/6)
senator_tweet_df['ari'] = senator_tweet_df['tweet_content'].apply(lambda s:textstat.automated_readability_index(s))
my_bar.progress(3/6)
senator_tweet_df['dale_chall'] = senator_tweet_df['tweet_content'].apply(lambda s:textstat.dale_chall_readability_score(s))
my_bar.progress(4/6)
senator_tweet_df['readability_consensus'] = senator_tweet_df['tweet_content'].apply(lambda s:textstat.text_standard(s))
my_bar.progress(5/6)
senator_tweet_df['reading_time'] = senator_tweet_df['tweet_content'].apply(lambda s:textstat.reading_time(s))
my_bar.progress(6/6)

# st.dataframe(data=senator_tweet_df)
# how can we integrate caching here 

# create a new df to display reading scores only 
st.title("Reading Scores")
reading_score_df = senator_tweet_df[['username', 'tweet_content', 'flesch', 'flesch_grade', 'ari', 'dale_chall', 'readability_consensus', 'reading_time']]
st.dataframe(data=reading_score_df)

# average reading score by politician 
st.title("Average Reading Scores by Politician")
mean_scores = senator_tweet_df.groupby(['username'])['flesch', 'flesch_grade', 'ari', 'dale_chall', 'readability_consensus', 'reading_time'].mean()
st.dataframe(data=mean_scores)

# topic modelling 

# Remove punctuation
reading_score_df['tweet_content'] = reading_score_df['tweet_content'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
reading_score_df['tweet_content']  = reading_score_df['tweet_content'].map(lambda x: x.lower())

# create a wordcloud 

# Join the different processed titles together.
long_string = ','.join(list(reading_score_df['tweet_content'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
st.image(wordcloud.to_image(), caption='Tweet Wordcloud')