# -*- coding: utf-8 -*-

import streamlit as st
from topics import TopicModel

import pandas as pd
import numpy as np

import heapq
import operator
import math

import itertools
import base64

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import graphviz as graphviz

from io import StringIO

from os import path, getcwd
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_corpus(url):
	return tm.load_corpus(url)

@st.cache(allow_output_mutation=True, persist=True, show_spinner=False)
def lda_model(url, stopwords, number_of_topics):
	corpus = load_corpus(url)
	with st.spinner("Training the topic model for {} topics ...".format(number_of_topics)):
		print("*** Training the topic model: {}".format(number_of_topics))
		if use_heuristic_alpha_value:
			return tm.fit(corpus, number_of_topics, alpha="talley")
		else:
			return tm.fit(corpus, number_of_topics)

# move this method to topics
def topics_to_csv(number_of_words):
	corpus = load_corpus(url)
	lda = lda_model(url, stopwords, number_of_topics)
	r = "topic, content\n"
	for index, topic in lda.show_topics(number_of_topics, number_of_words):
		line = "topic_{},".format(index)
		for w in topic:
			line += " " + w[0]
		r += line + "\n"
	return r

def read_topics(csv):
	return pd.read_csv(StringIO(csv))

def topics(number_of_words):
	return read_topics(topics_to_csv(number_of_words))

def download_link(dataframe, file_name, title="Download"):
	csv = dataframe.to_csv(index=False)
	download_link_from_csv(csv, file_name, title)

def download_link_from_csv(csv, file_name, title="Download"):
	b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
	href = "<a href='data:file/csv;base64,{}' download='{}'>{}</a>".format(b64, file_name, title)
	st.markdown(href, unsafe_allow_html=True)

def bow_top_keywords(bag_of_words, dictionary):
	keywords = []
	for wid, score in heapq.nlargest(3, bag_of_words, key=operator.itemgetter(1)):
		keywords.append("{}".format(dictionary[wid]))
	return keywords

def document_topics(i):
	corpus = load_corpus(url)
	lda = lda_model(url, stopwords, number_of_topics)
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
	lda = lda_model(url, stopwords, number_of_topics)
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
	lda = lda_model(url, stopwords, number_of_topics)
	return np.argsort(-np.array(topics_sparse_to_full(lda.get_document_topics)))	

# sum document frequencies for each topic and normalize
# thus, the column tallies add up to 1
def tally_columns(dtm):
	return [sum([row[k] for row in dtm])/len(dtm) for k in range(number_of_topics)]

def sort_by_topic(dtm, k, cut_off=0.80):
	col_k = [row[k] for row in dtm]
	top_documents_index = np.argsort(-np.array(col_k))
	return [index for index in top_documents_index 
		if dtm[index][k] >= cut_off]

# return a dictionary of topic keywords and their probabilities
def topic_words(k, number_of_words):
	r = {}
	corpus = load_corpus(url)
	lda = lda_model(url, stopwords, number_of_topics)
	for index, topic in lda.show_topics(number_of_topics, number_of_words):
		if index == k:
			for w in topic:
				s = w[0]
				r[s] = w[1]
			return r
	return {}

import re

def keyword_coocurrence_graph(selected_topic, min_edges, cut_off):
	corpus = load_corpus(url)
	dtm = document_topics_matrix()
	top_documents = sort_by_topic(dtm, selected_topic, cut_off)
	documents = corpus.documents['content'][top_documents]
	index = {}
	reverse_index = {}
	next_index = 0
	sentence_words = []
	for document in documents:
		for sentence in document.split(". "):
			sentence = re.sub(r'[^A-Za-z0-9]+', ' ', sentence)
			words = [word for word in sentence.lower().split(" ") 
				if word not in corpus.stopwords_en]
			words = set(words)
			for word in words:
				if word not in index:
					index[word] = next_index
					reverse_index[next_index] = word
					next_index = next_index + 1
			sentence_words.append(words)
	edge = np.zeros((len(index), len(index)))
	for words in sentence_words:
		for wi, wj in list(itertools.combinations(words, 2)):
			if wi < wj:
				edge[index[wi], index[wj]] = edge[index[wi], index[wj]] + 1
			else:
				edge[index[wj], index[wi]] = edge[index[wj], index[wi]] + 1
	graph = graphviz.Graph(format='png')
	graph.attr('node', shape='plaintext')
	nodes = []
	for i in range(len(index)):
		for j in range(len(index)):
			if edge[i, j] >= min_edges:
				if i not in nodes:
					nodes.append(i)
				if j not in nodes:
					nodes.append(j)
				graph.edge(reverse_index[i], reverse_index[j], 
					penwidth="{}".format(math.sqrt(edge[i, j])))
	for i in nodes:
		graph.node(reverse_index[i])
	return graph, [reverse_index[node] for node in nodes], top_documents

# experimental: create keyword co-occurrence graph from top keywords for a list
# of topics, instead of all non-stopword words in a document
def topic_keyword_coocurrence_graph(topic_range, min_edges, cut_off, topic_depth):
	# 1. build list of top documents and keywords for topics in topic_range
	corpus = load_corpus(url)
	dtm = document_topics_matrix()
	top_documents = []
	top_topic_keywords = []
	for topic in topic_range:
		# only include documents about the cut-off
		top_documents = top_documents + sort_by_topic(dtm, topic, cut_off)
		# todo: how can we do this in a way that topics with greater weight have more
		# of their keywords in this list that lower-weight topics?
		top_topic_keywords = top_topic_keywords + topic_words(topic, topic_depth).keys()
	documents = corpus.documents['content'][top_documents]

	# 2. extract keywords in top_topic_keywords from sentences in documents
	index = {}
	reverse_index = {}
	next_index = 0
	sentence_words = []
	for document in documents:
		for sentence in document.split(". "):
			sentence = re.sub(r'[^A-Za-z0-9]+', ' ', sentence)
			words = [word for word in sentence.lower().split(" ") 
				if word in top_topic_keywords]
			words = set(words)
			for word in words:
				if word not in index:
					index[word] = next_index
					reverse_index[next_index] = word
					next_index = next_index + 1
			sentence_words.append(words)

	# 3. create graph from all sentence-level co-occurrences of keywords
	edge = np.zeros((len(index), len(index)))
	for words in sentence_words:
		for wi, wj in list(itertools.combinations(words, 2)):
			if wi < wj:
				edge[index[wi], index[wj]] = edge[index[wi], index[wj]] + 1
			else:
				edge[index[wj], index[wi]] = edge[index[wj], index[wi]] + 1
	graph = graphviz.Graph(format='png')
	graph.attr('node', shape='plaintext')
	nodes = []
	for i in range(len(index)):
		for j in range(len(index)):
			if edge[i, j] >= min_edges:
				if i not in nodes:
					nodes.append(i)
				if j not in nodes:
					nodes.append(j)
				graph.edge(reverse_index[i], reverse_index[j], 
					penwidth="{}".format(math.sqrt(edge[i, j])))
	for i in nodes:
		graph.node(reverse_index[i])
	# return graph as well as list of nodes and range of documents
	return graph, [reverse_index[node] for node in nodes], top_documents

st.sidebar.title("Topic Model Explorer")
tm = TopicModel()

url = st.sidebar.file_uploader("Corpus", type="csv")

stopwords = st.sidebar.text_area("Stopwords (one per line)")
update_stopwords = st.sidebar.button("Update stopwords")

if update_stopwords:
	if url is not None:
		corpus = load_corpus(url)
		corpus.update_stopwords(stopwords)

show_documents = st.sidebar.checkbox("Show documents", value=True)

if show_documents:
	st.header("Corpus")
	if url is not None:
		corpus = load_corpus(url)
		if('name' not in corpus.documents or 'content' not in corpus.documents):
			st.markdown('''
		The corpus must have a *name* and a *content* column.
			''')
		st.dataframe(corpus.documents)
		download_link_from_csv("\n".join(corpus.stopwords), "stopwords.txt",
			"Download stopwords")
	else:
		st.markdown("Please upload a corpus.")

number_of_topics = st.sidebar.slider("Number of topics", 1, 50, 10)
use_heuristic_alpha_value = st.sidebar.checkbox("Use heuristic value for alpha", value=True)
show_topics = st.sidebar.checkbox("Show topics", value=False)

if show_topics:
	st.header("Topics")
	if url is not None:	
		corpus = load_corpus(url)	# needed for caching purposes (check)
		df = topics(5)
		st.table(df)
		if use_heuristic_alpha_value:
			st.markdown("Heuristic value of alpha (Talley et al., 2011): 0.05 (%.2f/%d) = %.2f" % (corpus.average_document_length(),
				number_of_topics, tm.alpha(corpus, number_of_topics)))
		download_link(df, "topic-keywords-{}.csv".format(number_of_topics),
			"Download topic keywords")
	else:
		st.markdown("No corpus.")

# show_correlation = st.sidebar.checkbox("Show correlation between topics and documents", value=False)
# if show_correlation:
# 	if url is not None:
# 		st.header("Correlation between topics and documents")
# 		corpus = load_corpus(url)
# 		st.markdown("Correlation for %d topics: %.2f" % 
# 			(number_of_topics, tm.cophenet(corpus, number_of_topics)))

show_topic_coherence = st.sidebar.checkbox("Show topic coherence", value=False)

if show_topic_coherence:
	st.header("Topic Coherence")
	if url is not None:	
		number_of_topics_range = st.sidebar.slider("Number of topics (range)", 1, 50, (5, 10))
		corpus = load_corpus(url)
		coherence_values = [lda_model(url, stopwords, num_topics).coherence(corpus)
			for num_topics in range(number_of_topics_range[0], number_of_topics_range[1] + 1)]
		t = range(number_of_topics_range[0], number_of_topics_range[1] + 1)
		plt.plot(t, coherence_values)
		plt.xlabel("Num Topics")
		plt.ylabel("Coherence Score")
		plt.show()
		st.pyplot()
	else:
		st.markdown("No corpus")

show_wordcloud = st.sidebar.checkbox("Show word cloud", value=False)

if show_wordcloud:
	st.header("Word cloud")
	if url is not None:
		selected_topic = st.sidebar.slider("Topic", 0, number_of_topics - 1, 0)
		number_of_words = st.sidebar.slider("Number of words", 1, 100, 25)
		st.markdown('''
			The word cloud shows the {} most frequent words for each topic.
		'''.format(number_of_words))
		mask = np.array(Image.open(path.join(getcwd(), "wordcloud-stencil.png")))
		wordcloud = WordCloud(background_color="white", mask=mask,
			max_font_size=32).fit_words(topic_words(selected_topic, number_of_words))
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.axis("off")
		plt.show()
		st.pyplot()
	else:
		st.markdown("No corpus.")

show_document_topic_matrix = st.sidebar.checkbox("Show document topics", value=False)

if show_document_topic_matrix:
	st.header("Document Topics")
	if url is not None:
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
		download_link(dtm_df, "document-topics-{}.csv".format(number_of_topics),
			"Download document topics")
	else:
		st.markdown("No corpus.")

show_tally_topics = st.sidebar.checkbox("Show topics tally", value=False)

if show_tally_topics:
	st.header("Topics Tally")
	if url is not None:
		st.markdown('''
			This graph show the proportion of each topic across the corpus.
		''')
		dtm = document_topics_matrix()
		topics = range(number_of_topics)
		plt.plot(topics, tally_columns(dtm))
		plt.xlabel("Num Topics")
		plt.ylabel("Tally")
		plt.show()
		st.pyplot()
	else:
		st.markdown("No corpus.")

show_topic_coocurrence_graph = st.sidebar.checkbox("Show topic co-occurrences", value=False)

if show_topic_coocurrence_graph:
	st.header("Topic Co-occurrences")
	if url is not None:
		min_weight = st.sidebar.slider("Minimum weight", 0.0, 0.5, value=0.1)
		min_edges = st.sidebar.slider("Minimum number of edges", 1, 10, value=1)
		st.markdown('''
			We consider topics to co-occur in the same document if the weight of both 
			topics for that document are greater than *minimum weight*. The thickness of
			an edge in the co-occurrance graph indicates how often two topics co-occur
			in a document (at least *minimum edges* times). Each node corresponds to a 
			topic. Node size represents the total weight of the topic.
		''')
		graph = topic_coocurrence_graph(min_weight, min_edges)
		st.graphviz_chart(graph)
	else:
		st.markdown("No corpus.")

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
	if url is not None:
		st.markdown('''
			This chart shows emerging topic trends. It plots the aggregated topic weights 
			and the contribution of each topic by year. Note: The corpus must have a *year*
			column. 
		''')
		dtm = document_topics_matrix()
		corpus = load_corpus(url)
		dtm_df = pd.DataFrame(dtm)
		if "year" in corpus.documents:
			dtm_df.insert(0, "year", [str(year) for year in corpus.documents["year"]])
			dtm_df_sum = dtm_df.groupby("year").sum()
			st.bar_chart(dtm_df_sum)
			dtm_df_sum_year = dtm_df_sum.copy()
			dtm_df_sum_year.insert(0, "year", sorted(dtm_df["year"].unique()))
			download_link(dtm_df_sum_year, "topic-trends-{}.csv".format(number_of_topics),
				"Download topic trends")
	else:
		st.markdown("No corpus.")

show_keyword_matches = st.sidebar.checkbox("Show keyword matches", value=False)
if show_keyword_matches:
	keywords = st.sidebar.text_input("Keywords")
	st.header("Keyword Matches")
	st.markdown('''
		Show which documents contain how many of the specified keywords.
	''')
	if url is not None and keywords != "":
		corpus = load_corpus(url)
		list_of_keywords = keywords.split(" ")
		df = corpus.documents.copy()
		for keyword in list_of_keywords:
			df[keyword] = [keyword in document.lower() for document
				in corpus.documents['content']]
		df['score'] = df[list_of_keywords].sum(axis=1)
		st.dataframe(df)
	else:
		st.markdown("No corpus or missing keywords.")

show_keyword_coocurrences = st.sidebar.checkbox("Show keyword co-occurrences", value=False)

if show_keyword_coocurrences:
	st.header("Keyword Co-occurrences")
	st.markdown('''
		Summarize the top documents in a given topic as a graph. 
		Its nodes are keywords in the documents (excluding language-specific, 
		but not user-defined stopwords), and its edges indicate that two 
		keywords appear in the same sentence. 
		The thickness of an edge indicates how often two keywords occur 
		together (at least *minimum edges* times). 
	''')
	if url is not None:
		keywords_selected_topic = st.sidebar.slider("Selected topic", 0, number_of_topics-1)
		keywords_cut_off = st.sidebar.slider("Minium topic weight", 0.0, 1.0, value=0.8)
		keywords_min_edges = st.sidebar.slider("Minimum number of edges", 1, 15, value=5)
		graph, nodes, top_docs = keyword_coocurrence_graph(keywords_selected_topic, 
			keywords_min_edges, keywords_cut_off)
		if len(nodes) == 0:
			st.markdown("No graph. Use less restrictive criteria.")
		else:
			st.graphviz_chart(graph)
			st.markdown("Top-ranked documents for topic {}:".format(keywords_selected_topic))
			corpus = load_corpus(url)
			st.dataframe(pd.DataFrame(corpus.documents).iloc[top_docs])
#		st.write(nodes)
	else:
		st.markdown("No corpus.")

show_topic_keyword_coocurrences = st.sidebar.checkbox("Show topic keyword co-occurrences (experimental)", value=False)

if show_topic_keyword_coocurrences:
	st.header("Topic Keyword Co-occurrences")
	st.markdown('''
		Summarize the top documents in a given topic as a graph. 
		Its nodes are keywords in the documents (including only the top topic
		keywords), and its edges indicate that two 
		keywords appear in the same sentence. 
		The thickness of an edge indicates how often two keywords occur 
		together (at least *minimum edges* times). 
	''')
	if url is not None:
		topic_keywords_selected_topic = st.sidebar.slider("Selected topic", 0, number_of_topics-1)
		topic_keywords_cut_off = st.sidebar.slider("Minium topic weight", 0.0, 1.0, value=0.8)
		topic_keywords_min_edges = st.sidebar.slider("Minimum number of edges", 1, 15, value=5)
		topic_keywords_topic_depth = st.sidebar.slider("Number of keywords per topic", 1, 50, value=10)
		graph, nodes, top_docs = topic_keyword_coocurrence_graph([topic_keywords_selected_topic], 
			topic_keywords_min_edges, topic_keywords_cut_off, topic_keywords_topic_depth)
		if len(nodes) == 0:
			st.markdown("No graph. Use less restrictive criteria.")
		else:
			st.graphviz_chart(graph)
			st.markdown("Top-ranked documents for topic {}:".format(topic_keywords_selected_topic))
			corpus = load_corpus(url)
			st.dataframe(pd.DataFrame(corpus.documents).iloc[top_docs])
#		st.write(nodes)
	else:
		st.markdown("No corpus.")

