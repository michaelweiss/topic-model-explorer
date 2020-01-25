import streamlit as st

import gensim as gs 
import gensim.parsing.preprocessing as pp

from gensim import models
from gensim.corpora import Dictionary

import pandas as pd

class TopicModel:
	def __init__(self):
		print("*** init TopicModel")
		self.corpus = None

	def gensim_version(self):
		return gs.__version__

	def load_corpus(self, url):
		documents = pd.read_csv(url)
		self.corpus = Corpus(documents)
		self.corpus.preprocess()

	def fit(self, corpus, number_of_topics):
		self.corpus = corpus
		self.number_of_topics = number_of_topics
		self.lda = models.LdaModel(corpus.bow(), number_of_topics)
		return self.lda

	def show_topics(number_of_words):
		return lda.show_topics(num_topics=number_of_topics, 
			num_words=number_of_words, formatted=False)

#	def get_document_topics(document):
#		return lda.

class Corpus:
	def __init__(self, docs):
		self.documents = docs
		self.tokens = []
		self.dictionary = Dictionary([])

	def preprocess(self):
		stopwords = self.read_stopwords('stopwords.txt')
		self.tokens = [[word for word in self.preprocess_document(document) 
				if word not in stopwords] 
			for document in self.documents['content']]
		self.dictionary = Dictionary(self.tokens)

	@st.cache
	def read_stopwords(self, file):
		file = open(file, 'r')
		return file.read().split('\n')

	def preprocess_document(self, document):
		return pp.strip_punctuation(document).lower().split()

	def bow(self):
		return [self.dictionary.doc2bow(doc) for doc in self.tokens]
