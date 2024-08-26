import os 
import numpy as np 
import pandas as pd 

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import nltk
nltk.download('wordnet')

df = pd.read_csv("data/Articles.csv", encoding='latin1')

stemmer = SnowballStemmer("english")
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = df['Article'].map(preprocess)
dictionary = gensim.corpora.Dictionary(processed_docs)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                       num_topics=10, 
                                       id2word = dictionary, 
                                       passes = 2, 
                                       workers=2)

for idx, topic in lda_model.print_topics(-1):
    print(f"Topic num:{idx} and words: {topic}","\n")

from gensim import corpora, models


tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]


lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, 
                                       num_topics=10, 
                                       id2word = dictionary, 
                                       passes = 2, 
                                       workers=2)


for idx, topic in lda_model_tfidf.print_topics(-1):
    print(f"Topic num:{idx} and words: {topic}","\n")

stemmer = SnowballStemmer("english")
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = df['processed_articles'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]


import numpy as np

from gensim import matutils
from gensim.models.ldamodel import LdaModel

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import gensim.corpora as corpora

def tokenize(text):
    #text = ''.join([ch for ch in text if ch not in string.punctuation and ch not in string.whitespace  and ch not in string.digits ])
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]




def fit_lda(corpus, id2word, num_topics=10, passes=20):
    return LdaModel(corpus, num_topics=num_topics,
                    passes=passes,
                    id2word=id2word)


def run(df, text_col, ngram_range, n_topics, n_passes):
    texts = df[text_col].values
    vec = CountVectorizer( stop_words='english', ngram_range = ngram_range, tokenizer=tokenize)
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names()
    
    corpus, id2word = matutils.Sparse2Corpus(X),  corpora.Dictionary(vec.inverse_transform(X))
    lda = fit_lda(corpus, id2word,  n_topics,n_passes )
    
    
    
    
    outputs = list(lda[corpus])
    df["topic"]=None
    for i,output in enumerate(outputs):
        topic =sorted(output,key=lambda x:x[1],reverse=True)[0][0]
        df["topic"][i] = topic
    
    return df, lda, corpus, vocab, id2word, vec, X



import os
import pyLDAvis.gensim
import pickle 
import pyLDAvis
pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join('./results/new'+str(n_topics))

LDAvis_prepared = pyLDAvis.gensim.prepare(lda, corpus, id2word)
with open(LDAvis_data_filepath, 'wb') as f:
    pickle.dump(LDAvis_prepared, f)

with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(n_topics) +'.html')




