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


nlp = spacy.load('en_core_web_lg')


st.title('Tweet Scraper')

# read tweets from before 
senator_tweet_df = pd.read_csv('C:/Users/Admin/Documents/GitHub/politics_project/senator_tweets_df.csv')

st.dataframe(data=senator_tweet_df)