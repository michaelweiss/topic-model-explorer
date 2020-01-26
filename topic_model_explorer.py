import streamlit as st
from topics import TopicModel

import pandas as pd
import numpy as np

import heapq
import operator
import math
import itertools

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import graphviz as graphviz

from io import StringIO

@st.cache(allow_output_mutation=True)
def load_corpus(url):
	print("*** Loading the corpus: {}".format(url))
	return tm.load_corpus(url)

@st.cache(allow_output_mutation=True)
def lda_model(url, number_of_topics):
	corpus = load_corpus(url)
	with st.spinner("Training the topic model ..."):
		print("*** Training the topic model: {}".format(number_of_topics))
		lda = tm.fit(corpus, number_of_topics)
		print("*** Training completed ***")
		return lda

# move this method to topics
def topics_to_csv(number_of_words):
	corpus = load_corpus(url)
	lda = lda_model(url, number_of_topics)
	r = "topic, content\n"
	for index, topic in lda.show_topics(number_of_topics, number_of_words):
		line = "topic_{},".format(index)
		for w in topic:
			line += " " + corpus.dictionary[int(w[0])]
		r += line + "\n"
	return r

def read_topics(csv):
	return pd.read_csv(StringIO(csv))

def topics(number_of_words):
	return read_topics(topics_to_csv(number_of_words))

def bow_top_keywords(bag_of_words, dictionary):
	keywords = []
	for wid, score in heapq.nlargest(3, bag_of_words, key=operator.itemgetter(1)):
		keywords.append("{}".format(dictionary[wid]))
	return keywords

def document_topics(i):
	corpus = load_corpus(url)
	lda = lda_model(url, number_of_topics)
	return lda.get_document_topics(corpus.documents[i])
	# return [bow_top_keywords(document, dictionary) for document in corpus]
	# return lda[corpus[i]]

def topics_sparse_to_full(topics):
	topics_full = [0] * number_of_topics  # pythonic way of creating a list of zeros
	for topic, score in topics:
		topics_full[topic] = score
	return topics_full

def document_topics_matrix():
	corpus = load_corpus(url)
	lda = lda_model(url, number_of_topics)
	dtm = []
	for document_bow in corpus.bow():
		dtm.append(topics_sparse_to_full(lda.get_document_topics(document_bow)))
	return dtm

def topic_coocurrence_matrix_(min_weight):
	dtm = document_topics_matrix()
	relationships = []
	for topic_weights in dtm:
		document_relationships = []
		for k in range(number_of_topics):
			if topic_weights[k] >= min_weight:
				document_relationships.append(k)
		relationships.append(document_relationships)
	return relationships

def topic_coocurrence_graph(min_weight, min_edges):
	dtm = document_topics_matrix()
	graph = graphviz.Graph()
	graph.attr('node', shape='circle', fixedsize='true')
	total_topic_weights = tally_columns(dtm)
	for i in range(number_of_topics):
		graph.node(str(i), width=str(2*math.sqrt(total_topic_weights[i])))
	edge = np.zeros((number_of_topics, number_of_topics))
	for topic_weights in dtm:
		topics = [k for k in range(number_of_topics) if topic_weights[k] >= min_weight]
		for i, j in list(itertools.combinations(topics, 2)):
			edge[i, j] = edge[i, j] + 1
	for i in range(number_of_topics):
		for j in range(number_of_topics):
			if edge[i, j] >= min_edges:
				graph.edge(str(i), str(j), 
					penwidth="{}".format(edge[i, j]))
	return graph

def normalize(df):
	df_new = df.copy()
	for topic in df.columns:
		topic_sum = df[topic].sum()
		df_new[topic] = df[topic]/topic_sum
	return df_new

def document_top_topics(i):
	lda = lda_model(url, number_of_topics)
	return np.argsort(-np.array(topics_sparse_to_full(lda.get_document_topics)))	

# sum document frequencies for each topic and normalize
# thus, the column tallies add up to 1
def tally_columns(dtm):
	return [sum([row[k] for row in dtm])/len(dtm) for k in range(number_of_topics)]

def sort_by_topic(dtm, k):
	col_k = [row[k] for row in dtm]
	return np.argsort(-np.array(col_k))

def topic_words(k, number_of_words):
	r = {}
	corpus = load_corpus(url)
	lda = lda_model(url, number_of_topics)
	for index, topic in lda.show_topics(number_of_topics, number_of_words):
		if index == k:
			for w in topic:
				s = corpus.dictionary[int(w[0])]
				r[s] = w[1]
			return r
	return {}

st.sidebar.title("Topic Model Explorer")
tm = TopicModel()

url = st.sidebar.text_input("Corpus (URL to a CSV file)", "abstracts.csv")
show_documents = st.sidebar.checkbox("Show documents", value=True)

if show_documents:
	st.header("Corpus")
	corpus = load_corpus(url)
	st.dataframe(corpus.documents)

number_of_topics = st.sidebar.slider("Number of topics", 1, 50, 10)
show_topics = st.sidebar.checkbox("Show topics", value=True)

if show_topics:
	st.header("Topics")
	st.table(topics(5))

show_wordcloud = st.sidebar.checkbox("Show word cloud", value=False)

if show_wordcloud:
	selected_topic = st.sidebar.slider("Topic", 0, number_of_topics - 1, 0)
	st.header("Word cloud")
	st.markdown('''
		The word cloud shows the 10 most frequent words for each topic.
	''')
	wordcloud = WordCloud(background_color="white", 
		max_font_size=28).fit_words(topic_words(selected_topic, 10))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()
	st.pyplot()

show_document_topic_matrix = st.sidebar.checkbox("Show document topics", value=False)

if show_document_topic_matrix:
	st.header("Document Topics")
	st.markdown('''
		The document topic matrix shows the topic weights for each document. 
	''')
	dtm = document_topics_matrix()
	corpus = load_corpus(url)
	dtm_df = pd.DataFrame(dtm)
	if "year" in corpus.documents:
		dtm_df.insert(0, "year", corpus.documents["year"])
	dtm_df.insert(0, "name", corpus.documents["name"])
	st.dataframe(dtm_df)

show_tally_topics = st.sidebar.checkbox("Show topics tally", value=False)

if show_tally_topics:
	st.header("Topics Tally")
	st.markdown('''
		This graph show the proportion of each topic across the corpus.
	''')
	dtm = document_topics_matrix()
	st.line_chart(tally_columns(dtm))

show_topic_coocurrence_graph = st.sidebar.checkbox("Show topic co-occurrences", value=False)

if show_topic_coocurrence_graph:
	min_weight = st.sidebar.slider("Minimum weight", 0.0, 0.3, value=0.1)
	min_edges = st.sidebar.slider("Minimum number of edges", 1, 10, value=1)
	st.header("Topic Co-occurrences")
	st.markdown('''
		We consider topics to co-occur in the same document if the weight of both 
		topics for that document are greater than *minimum weight*. The thickness of
		a line in the co-occurrance graph indicates how often two topics co-occur
		in a document (at least *minimum edges* times). Each node corresponds to a 
		topic. Node size represents the total weight of the topic.
	''')
	graph = topic_coocurrence_graph(min_weight, min_edges)
	st.graphviz_chart(graph)

# show_sorted_topics = st.sidebar.checkbox("Show sorted topics", value=False)

# if show_sorted_topics:
# 	selected_topic = st.sidebar.slider("Topic", 0, number_of_topics - 1, 0)
# 	st.header("Sorted topics")
# 	dtm = document_topics_matrix()
# 	# st.table(topics_d[sort_by_topic(dtm, 0)])
# 	st.table(sort_by_topic(dtm, selected_topic))

show_topic_trends = st.sidebar.checkbox("Show topics trends", value=False)

if show_topic_trends:
	st.header("Topic Trends")
	st.markdown('''
		This chart shows emerging topic trends. It plots the aggregated topic weights 
		and the contribution of each topic by year. Note: The corpus must have a *year*
		column. 
	''')
	dtm = document_topics_matrix()
	corpus = load_corpus(url)
	dtm_df = pd.DataFrame(dtm)
	dtm_df.insert(0, "year", corpus.documents["year"])
	dtm_df_sum = dtm_df.groupby("year").sum()
	st.bar_chart(dtm_df_sum)

