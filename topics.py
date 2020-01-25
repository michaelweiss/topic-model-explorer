import streamlit as st

import gensim as gs 
import gensim.parsing.preprocessing as pp

from gensim import models
from gensim.corpora import Dictionary

import pandas as pd

class TopicModel:
	def gensim_version(self):
		return gs.__version__

	def load_corpus(self, url):
		documents = pd.read_csv(url)
		corpus = Corpus(documents)
		corpus.preprocess()
		return corpus

	def fit(self, corpus, number_of_topics):
		self.corpus_ = corpus
		self.number_of_topics_ = number_of_topics
		self.lda_ = models.LdaModel(corpus.bow(), number_of_topics)
		return self.lda_


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
