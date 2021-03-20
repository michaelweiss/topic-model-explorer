# -*- coding: utf-8 -*-

import streamlit as st

import gensim as gs

from gensim import models
from gensim.models.coherencemodel import CoherenceModel
# from gensim.models import ldamulticore
from gensim.corpora import Dictionary

import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import math

from io import StringIO
from re import sub

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import MWETokenizer

class TopicModel:
	def gensim_version(self):
		return gs.__version__

	def load_corpus(self, url, stopwords, multiwords):
		if url is not None:
			url.seek(0)	 # move read head back to the start (StringIO behaves like a file)
			documents = pd.read_csv(url)
			corpus = Corpus(documents)
			corpus.preprocess(stopwords, multiwords)
			return corpus
		else:
			return None

	def fit(self, corpus, number_of_topics, number_of_iterations=50, number_of_passes=1,
			number_of_chunks=1, alpha="symmetric"):
		if alpha == "talley":
			alpha = np.array([self.alpha(corpus, number_of_topics)] * number_of_topics)
		return LDA(models.LdaModel(corpus.bow(), number_of_topics, corpus.dictionary,
			iterations=number_of_iterations, passes=number_of_passes, 
			chunksize=self.chunksize(corpus, number_of_chunks), alpha=alpha))

	def alpha(self, corpus, number_of_topics):
		return 0.05 * corpus.average_document_length() / number_of_topics

	def chunksize(self, corpus, number_of_chunks):
		return math.ceil(len(corpus.documents) / number_of_chunks)

class LDA:
	def __init__(self, lda):
		self.lda = lda

	def number_of_topics(self):
		return self.lda.num_topics

	def chunksize(self):
		return self.lda.chunksize

	def show_topics(self, number_of_topics, number_of_words):
		return self.lda.show_topics(num_topics=number_of_topics, 
			num_words=number_of_words, formatted=False)

	def get_document_topics(self, document_bow):
		return self.lda.get_document_topics(document_bow)

	def coherence(self, corpus):
		coherence_model = CoherenceModel(model=self.lda, texts=corpus.tokens, 
			dictionary=corpus.dictionary, coherence='c_uci')
		return coherence_model.get_coherence()

	# return a difference matrix between two topic models
	# computes the average jaccard distance as defined by Greene (2014)
	def difference(self, other, n=10):
		return sum([self.jaccard(other, k) for k in range(n)]) / n

	def jaccard(self, other, k):
		diff, _ = self.lda.diff(other.lda, distance='jaccard', num_words=k)
		return diff

	def document_topics_matrix(self, corpus):
		dtm = []
		for document_bow in corpus.bow():
			dtm.append(self.topics_sparse_to_full(self.get_document_topics(document_bow)))
			tcid = corpus.dictionary.id2token
		return pd.DataFrame(dtm)

	def topics_sparse_to_full(self, topics):
		topics_full = [0] * self.number_of_topics()  # pythonic way of creating a list of zeros
		for topic, score in topics:
			topics_full[topic] = score
		return topics_full

class TopicAlignment:
	def __init__(self, topic_model, corpus, number_of_topics, number_of_chunks, number_of_runs):
		self.topic_model = topic_model
		self.corpus = corpus
		self.number_of_topics = number_of_topics
		self.number_of_chunks = number_of_chunks
		self.number_of_runs = number_of_runs

	def fit(self, progress_update):
		# experimental: remember the computed LDA models
		# don't store the computed LDA models; in this way, they don't get 
		# included in the hash streamlit uses to cache results
		self.lda_models = self.lda_model_runs(progress_update)
		# determine the matching topics across the different runs
		self.matches = self.matches(self.lda_models)
		# find the top topic keywords for each topic and each run
		self.topics = self.topics(self.lda_models)
		# collect the keywords and associated weights for each topic across all topic models
		self.keywords, self.weights = self.keywords_with_weights(self.lda_models)
		# find the topics for each document
		self.dtm, self.documents = self.documents(self.lda_models)

	# create a group of topic models with the same number of topics
	def lda_model_runs(self, progress_update):
		lda_models = []
		for run in range(self.number_of_runs):
			lda_models.append(self.topic_model.fit(self.corpus, self.number_of_topics, 
				number_of_chunks=self.number_of_chunks))
			progress_update(run)
		return lda_models

	# extract the topic words for each topic in all topic models
	def topics(self, lda_models):
		return pd.DataFrame([[" ".join([tw[0] for tw in lda_model.lda.show_topic(t, 10)]) 
			for lda_model in lda_models] for t in range(self.number_of_topics)])

	# compute the average Jaccard distance between the topic models
	def differences(self, lda_models):
		return [lda_models[0].difference(lda_models[i]) 
			for i in range(1, len(lda_models))]

	# fit topics between the first and each of the remaining topic models using
	# the Hungarian linear assignment method
	def matches(self, lda_models):
		diffs = self.differences(lda_models)
		matches = pd.DataFrame()
		# first column are the topics of the first topic model
		matches[0] = range(self.number_of_topics)
		# minimize the total misalignment between topics
		for i in range(1, self.number_of_runs):
			_, cols = linear_sum_assignment(diffs[i-1])
			# each column contains the topics that align with the topics of the first topic
			matches[i] = cols
		return matches

	def keywords_with_weights(self, lda_models):
		keywords, weights = [], []
		for topic in range(self.number_of_topics):
			keywords_for_topic = pd.DataFrame()
			weights_for_topic = pd.DataFrame()
			for i in range(self.number_of_runs):
				keywords_for_topic[i] = [tw[0] for tw 
					in lda_models[i].lda.show_topic(self.matches[i][topic], 10)]
				weights_for_topic[i] = [tw[1] for tw 
					in lda_models[i].lda.show_topic(self.matches[i][topic], 10)]
			keywords.append(keywords_for_topic)
			weights.append(weights_for_topic)
		return keywords, weights

	def documents(self, lda_models):
		dtm = [lda_models[i].document_topics_matrix(self.corpus)
			for i in range(self.number_of_runs)]
		documents = []
		for topic in range(self.number_of_topics):
			documents_for_topic = pd.DataFrame()
			for i in range(self.number_of_runs):
				documents_for_topic[i] = dtm[i][self.matches[i][topic]]
			documents.append(documents_for_topic)
		return dtm, documents

class Corpus:
	def __init__(self, documents):
		self.documents = self.to_ascii(documents)

	def to_ascii(self, documents):
		# replace non-ascii symbols left by text processing software
		documents['content'] = [sub(r'[^A-Za-z0-9,\.?!]+', ' ', document)
			for document in documents['content']]
		return documents

	def preprocess(self, user_defined_stopwords, multiwords):
		self.stopwords_en = self.read_stopwords("stopwords-en.txt")
		self.user_defined_stopwords = user_defined_stopwords.split('\n')
		self.user_defined_stopwords = [word.strip() for word in self.user_defined_stopwords]
		self.stopwords = self.stopwords_en + self.user_defined_stopwords
		tokenizer = self.create_tokenizer(multiwords)
		self.tokens = [[word for word in tokenizer.tokenize([self.lemmatize(word) for word in self.tokenize(document)])
				if word not in self.stopwords]
			for document in self.documents['content']]
		# self.tokens = [word for word in [[self.lemmatize(word) for word in self.tokenize(document)]
		# 	for document in self.documents['content']]]
		# self.tokens = [tokenizer.tokenize(word_list) for word_list in self.tokens]
		# self.tokens = [word for word in word_list if word not in self.stopwords]
		self.dictionary = Dictionary(self.tokens)

	def read_stopwords(self, file):
		file = open(file, 'r')
		return file.read().split('\n')

	def create_tokenizer(self, multiwords):
		tokenizer = MWETokenizer()
		for mwe in tokenizer.tokenize(multiwords.split('\n')):
			tokenizer.add_mwe(mwe.split(' '))
		return tokenizer

	def tokenize(self, document):
		return sub(r'[^A-Za-z0-9]+', ' ', document).lower().split()

	def lemmatize(self, word):
		lemmatizer = WordNetLemmatizer()
		return lemmatizer.lemmatize(word)

	def bow(self):
		return [self.dictionary.doc2bow(doc) for doc in self.tokens]

	def average_document_length(self):
		return np.mean(map(len, self.tokens))

# initialize

nltk.download('wordnet') 

