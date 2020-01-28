import streamlit as st

import gensim as gs 
import gensim.parsing.preprocessing as pp

from gensim import models
from gensim.corpora import Dictionary

import pandas as pd
from io import StringIO
import re

class TopicModel:
	def gensim_version(self):
		return gs.__version__

	def load_corpus(self, url):
		if url is not None:
			print("*** Reading the corpus")
			documents = pd.read_csv(url)
			corpus = Corpus(documents)
			corpus.preprocess()
			return corpus
		else:
			print("*** No url provided")
			corpus = Corpus([])	# exception
			return corpus

	def fit(self, corpus, number_of_topics):
		return LDA(models.LdaModel(corpus.bow(), number_of_topics))

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
		documents['content'] = [re.sub(r'[^A-Za-z0-9,\.?!]+', ' ', document)
			for document in documents['content']]
		return documents

	def preprocess(self):
		self.tokens = [[word for word in self.preprocess_document(document) 
				if word not in self.stopwords] 
			for document in self.documents['content']]
		self.dictionary = Dictionary(self.tokens)

	def initialize_stopwords(self):
		self.read_stopwords("stopwords-en.txt")

	def read_stopwords(self, file):
		file = open(file, 'r')
		self.stopwords = file.read().split('\n')

	def update_stopwords(self, stopwords):
		print("*** Update stopwords: {}".format(stopwords.split('\n')))
		new_stopwords = stopwords.split('\n')
		self.initialize_stopwords()
		self.stopwords = self.stopwords + new_stopwords
		# update tokens after adding stopwords
		self.preprocess()

	def preprocess_document(self, document):
		return pp.strip_punctuation(document).lower().split()

	def bow(self):
		return [self.dictionary.doc2bow(doc) for doc in self.tokens]
