# -*- coding: utf-8 -*-

import streamlit as st

import gensim as gs 

from gensim import models
# from gensim.models import ldamulticore
from gensim.corpora import Dictionary

import pandas as pd

from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer

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
			corpus = Corpus([])	# exception
			return corpus

	def fit(self, corpus, number_of_topics):
		return LDA(models.LdaModel(corpus.bow(), number_of_topics, corpus.dictionary))

	# Based on a code snippet from the chapter "Textsammlung. Ein Beispiel aus der 
	# Geschichte der Soziologie" in Papilloud (2018), Qualitative Textanalyse mit Topic-Modellen: 
	# Eine Einführung für Sozialwissenschaftler
	def cophenet(self, corpus, number_of_topics):
		tf_matrix = corpus.tf_matrix()
		nmf = decomposition.NMF(n_components=number_of_topics, random_state=1)
		doctopic = nmf.fit_transform(tf_matrix)
		linkage_doc = linkage(doctopic, 'ward')
		coph_corr = cophenet(linkage_doc, pdist(doctopic))
		return coph_corr[0]

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

class LDA:
	def __init__(self, lda):
		self.lda = lda

	def show_topics(self, number_of_topics, number_of_words):
		return self.lda.show_topics(num_topics=number_of_topics, 
			num_words=number_of_words, formatted=False)

	def get_document_topics(self, document_bow):
		return self.lda.get_document_topics(document_bow)

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

	def tf_matrix(self):
		vectorize = TfidfVectorizer(min_df=2, max_df=0.95, encoding='utf-8', 
			sublinear_tf='True', analyzer='word', ngram_range=(1,1), stop_words = self.stopwords)
		return vectorize.fit_transform(self.documents['content'])

