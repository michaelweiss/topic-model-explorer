# -*- coding: utf-8 -*-

import streamlit as st

import gensim as gs 

from gensim import models
from gensim.models.coherencemodel import CoherenceModel
# from gensim.models import ldamulticore
from gensim.corpora import Dictionary

import pandas as pd
import numpy as np
import math

from io import StringIO
from re import sub

class TopicModel:
	def gensim_version(self):
		return gs.__version__

	def load_corpus(self, url):
		if url is not None:
			url.seek(0)	# move read head back to the start (StringIO behaves like a file)
			documents = pd.read_csv(url)
			corpus = Corpus(documents)
			corpus.preprocess()
			return corpus
		else:
			print("*** No url provided")
			corpus = Corpus(pd.DataFrame({
				'name': [],
				'content': []
			}))	 # exception
			return corpus

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

	# def show_topics(self, number_of_words):
	# 	return self.lda.show_topics(num_topics=self.number_of_topics, 
	# 		num_words=number_of_words, formatted=False)

	# def get_document_topics(document):
	# return lda.

	# def topics_to_csv(self, number_of_words):
	# 	print("*** TopicModel.Topics to csv")
	# 	print(self.lda)
	# 	r = "topic, content\n"
	# 	for index, topic in self.show_topics(number_of_words):
	# 		line = "topic_{},".format(index)
	# 		for w in topic:
	# 			line += " " + self.corpus.dictionary[int(w[0])]
	# 		r += line + "\n"
	# 	return r

	# def read_topics(self, csv):
	# 	return pd.read_csv(StringIO(csv))

	# def topics(self, number_of_words):
	# 	return self.read_topics(self.topics_to_csv(number_of_words))

# TODO: move logic for running multiple topic models from topic-model-explorer
class TopicModelRuns:
	def __init__(self, topic_model, url, stopwords, number_of_chunks, number_of_topics, n=5):
		self.topic_model = topic_model
		self.url = url
		self.stopwords = stopwords
		self.number_of_chunks = number_of_chunks
		self.number_of_topics = number_of_topics
		self.n = n
		# run_topic_models()

class LDA:
	def __init__(self, lda):
		self.lda = lda

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

class Corpus:
	def __init__(self, documents):
		self.documents = self.to_ascii(documents)
		self.initialize_stopwords()

	def to_ascii(self, documents):
		# replace non-ascii symbols left by text processing software
		documents['content'] = [sub(r'[^A-Za-z0-9,\.?!]+', ' ', document)
			for document in documents['content']]
		return documents

	def preprocess(self):
		self.tokens = [[word for word in sub(r'[^A-Za-z0-9]+', ' ', document).lower().split() 
				if word not in self.stopwords] 
			for document in self.documents['content']]
		self.dictionary = Dictionary(self.tokens)

	def initialize_stopwords(self):
		self.read_stopwords("stopwords-en.txt")

	def read_stopwords(self, file):
		file = open(file, 'r')
		self.stopwords = file.read().split('\n')
		self.stopwords_en = self.stopwords

	def update_stopwords(self, stopwords):
		print("*** Update stopwords: {}".format(stopwords.split('\n')))
		new_stopwords = stopwords.split('\n')
		self.initialize_stopwords()
		self.stopwords = self.stopwords + new_stopwords
		# update tokens after adding stopwords
		self.preprocess()

	def bow(self):
		return [self.dictionary.doc2bow(doc) for doc in self.tokens]

	def average_document_length(self):
		return np.mean(map(len, self.tokens))

