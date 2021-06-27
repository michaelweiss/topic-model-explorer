# -*- coding: utf-8 -*-

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import math
import itertools
import base64
import re
import graphviz as graphviz
from pyvis.network import Network

from topics import TopicModel, LDA

# model

@st.cache(allow_output_mutation=True)
def load_corpus(url, stopwords, multiwords):
	return tm.load_corpus(url, stopwords, multiwords)

@st.cache(hash_funcs={LDA: id})
def topic_model(corpus, number_of_topics, number_of_chunks):
	return tm.fit(corpus, number_of_topics, number_of_chunks=number_of_chunks)

def topics(model):
	return pd.DataFrame([[" ".join([tw[0] for tw in model.lda.show_topic(t, 10)])] 
		for t in range(number_of_topics)])

def document_topic_matrix(model, corpus):
	return model.document_topic_matrix(corpus)

# sum document frequencies for each topic and normalize
# thus, the column tallies add up to 1
def tally_columns(dtm, number_of_topics):
	return [sum([row[k] for row in dtm])/len(dtm) for k in range(number_of_topics)]

def topic_coocurrence_graph(model, corpus, number_of_topics, min_weight, min_edges):
	dtm = document_topic_matrix(model, corpus).to_numpy()
	keywords = ["\n".join([tw[0] for tw in model.lda.show_topic(t, 3)])
		for t in range(number_of_topics)]
	graph = graphviz.Graph()
	graph.attr('node', shape='circle', fixedsize='true')
	total_topic_weights = tally_columns(dtm, number_of_topics)
	for i in range(number_of_topics):
		graph.node(str(i), width=str(4*math.sqrt(total_topic_weights[i])), label=keywords[i])
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

def topic_coocurrence_graph_pyvis(model, corpus, number_of_topics, min_weight, min_edges, smooth_edges):
	dtm = document_topic_matrix(model, corpus).to_numpy()
	keywords = ["\n" + "\n".join([tw[0] for tw in model.lda.show_topic(t, 3)])
		for t in range(number_of_topics)]
	graph = Network("600px", "100%", notebook=True, heading='')
	total_topic_weights = tally_columns(dtm, number_of_topics)
	for i in range(number_of_topics):
		graph.add_node(i, label=keywords[i], size=4*10*math.sqrt(total_topic_weights[i]),
			title="Topic {}".format(i))
	edge = np.zeros((number_of_topics, number_of_topics))
	for topic_weights in dtm:
		topics = [k for k in range(number_of_topics) if topic_weights[k] >= min_weight]
		for i, j in list(itertools.combinations(topics, 2)):
			edge[i, j] = edge[i, j] + 1
	for i in range(number_of_topics):
		for j in range(number_of_topics):
			if edge[i, j] >= min_edges:
				graph.add_edge(i, j, value=edge[i, j], smooth=smooth_edges)
	return graph

def keyword_coocurrence_graph(model, corpus, selected_topic, min_edges, cut_off):
	# step 1: select most relevant documents for the selected topic
	dtm = document_topic_matrix(model, corpus).to_numpy()
	top_documents = sort_by_topic(dtm, selected_topic, cut_off)
	documents = corpus.documents['content'][top_documents]

	# step 2: parse the content of the documents and extract the unique words from each sentence
	index = {}
	reverse_index = {}
	next_index = 0
	sentence_words = []
	for document in documents:
		for sentence in re.split('[?!.]', document):
			sentence = re.sub(r'[^A-Za-z0-9]+', ' ', sentence)
			words = [word for word in sentence.lower().split(" ") 
				if word not in corpus.stopwords]
			words = set(words)
			for word in words:
				if word not in index:
					index[word] = next_index
					reverse_index[next_index] = word
					next_index = next_index + 1
			sentence_words.append(words)

	# step 3: count the number of word co-occurrences
	edge = np.zeros((len(index), len(index)))
	for words in sentence_words:
		for wi, wj in list(itertools.combinations(words, 2)):
			if wi < wj:
				edge[index[wi], index[wj]] = edge[index[wi], index[wj]] + 1
			else:
				edge[index[wj], index[wi]] = edge[index[wj], index[wi]] + 1

	# step 4: create a word co-occurrence network
	graph = Network("600px", "100%", notebook=True, heading='')
	nodes = []
	for i in range(len(index)):
		for j in range(len(index)):
			if edge[i, j] >= min_edges:
				if i not in nodes:
					nodes.append(i)
				if j not in nodes:
					nodes.append(j)	
	for i in nodes:
		graph.add_node(i, reverse_index[i], size=10)
	for i in range(len(index)):
		for j in range(len(index)):
			if edge[i, j] >= min_edges:
				graph.add_edge(i, j, value=math.sqrt(edge[i, j]), smooth=True)

	return graph, [reverse_index[node] for node in nodes], top_documents
	
def sort_by_topic(dtm, k, cut_off=0.80):
	col_k = [row[k] for row in dtm]
	top_documents_index = np.argsort(-np.array(col_k))
	return [index for index in top_documents_index 
		if dtm[index][k] >= cut_off]

# view

def show_documents(corpus):
	st.header("Documents")
	if corpus is not None:
		if st.checkbox("Show table with full text", value=False):
			st.table(corpus.documents)
		else:
			st.dataframe(corpus.documents, height=150)
		download_link_from_csv("\n".join(corpus.stopwords), "stopwords.txt",
			"Download stopwords")
	else:
		st.markdown("No corpus loaded, or missing the expected *name* and *content* columns")

def show_topics(corpus, number_of_topics, number_of_chunks=100):
	st.header("Topics")
	if corpus is None:
		st.markdown("Please upload a corpus first")
	else:
		topics_df = topics(topic_model(corpus, number_of_topics, number_of_chunks))
		st.table(topics_df)
		download_link(topics_df, "topic-keywords-{}.csv".format(number_of_topics),
			"Download topic keywords")

def show_document_topic_matrix(corpus, number_of_topics, number_of_chunks=100):
	st.header("Document topic matrix")
	if corpus is None:
		st.markdown("Please upload a corpus first")
	else:
		dtm_df = document_topic_matrix(topic_model(corpus, number_of_topics, number_of_chunks), corpus)
		if "year" in corpus.documents:
			dtm_df.insert(0, "year", corpus.documents["year"])
		dtm_df.insert(0, "name", corpus.documents["name"])
		st.dataframe(dtm_df, height=150)
		download_link(dtm_df, "document-topic-matrix-{}.csv".format(number_of_topics),
			"Download document topic matrix")

def show_topic_co_occurrences(corpus, number_of_topics, number_of_chunks=100):
	st.header("Topic co-occurrences")
	if corpus is None:
		st.markdown("Please upload a corpus first")
	else:
		with st.beta_expander("Help"):
			st.markdown('''
				We consider topics to co-occur in the same document if the weight of both 
				topics for that document are greater than *minimum weight*. The thickness of
				an edge in the co-occurrance graph indicates how often two topics co-occur
				in a document (at least *minimum edges* times). Each node represents a 
				topic. Node size reflects the total weight of the topic.
			''')
		min_weight = st.sidebar.slider("Minimum weight", 0.0, 0.5, value=0.1, step=0.05)
		min_edges = st.sidebar.slider("Minimum number of edges", 1, 10, value=1)
		if st.sidebar.radio("Visualization library to use", ("VisJS", "GraphViz"), index=0) == "VisJS":
			smooth_edges = st.sidebar.checkbox("Draw with smooth edges", value=False)
			graph_pyvis = topic_coocurrence_graph_pyvis(topic_model(corpus, number_of_topics, number_of_chunks), 
				corpus, number_of_topics, min_weight, min_edges, smooth_edges)
			graph_pyvis.show("topic-graph.html")
			components.html(open("topic-graph.html", 'r', encoding='utf-8').read(), height=625)
		else:
			graph = topic_coocurrence_graph(topic_model(corpus, number_of_topics, number_of_chunks), 
				corpus, number_of_topics, min_weight, min_edges)
			st.graphviz_chart(graph)

def show_keyword_co_coccurrences(corpus, number_of_topics, number_of_chunks):
	st.header("Keyword co-occurrences")
	if corpus is None:
		st.markdown("Please upload a corpus first")
	else:
		with st.beta_expander("Help"):
			st.markdown('''
				Summarize the top documents in a given topic as a graph. 
				Its nodes are keywords in the documents (excluding language-specific, 
				but not user-defined stopwords), and its edges indicate that two 
				keywords appear in the same sentence. 
				The thickness of an edge indicates how often two keywords occur 
				together (at least *minimum edges* times). 
			''')
		keywords_selected_topic = st.sidebar.slider("Selected topic", 0, number_of_topics-1)
		keywords_cut_off = st.sidebar.slider("Minium topic weight", 0.0, 1.0, value=0.8, step=0.05)
		keywords_min_edges = st.sidebar.slider("Minimum number of edges", 1, 15, value=5)
		graph, nodes, top_documents = keyword_coocurrence_graph(topic_model(corpus, number_of_topics, number_of_chunks), corpus, 
			keywords_selected_topic, keywords_min_edges, keywords_cut_off)
		if len(nodes) == 0:
			st.markdown("No graph. Use less restrictive criteria.")
		else:
			graph.show("keyword-graph.html")
			components.html(open("keyword-graph.html", 'r', encoding='utf-8').read(), height=625)

# view helpers

def download_link_from_csv(csv, file_name, title="Download"):
	b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
	href = "<a href='data:file/csv;base64,{}' download='{}'>{}</a>".format(b64, file_name, title)
	st.markdown(href, unsafe_allow_html=True)

def download_link(dataframe, file_name, title="Download"):
	csv = dataframe.to_csv(index=False)
	download_link_from_csv(csv, file_name, title)

# controller

tm = TopicModel()

st.sidebar.title("Topic Model Explorer")
st.sidebar.write("Uses [streamlit](https://streamlit.io) {} and [gensim](https://radimrehurek.com/gensim/) {}".format(st.__version__, tm.gensim_version()))

url = st.sidebar.file_uploader("Corpus", type="csv")
stopwords = st.sidebar.text_area("Stopwords (one per line)")
multiwords = st.sidebar.text_area("Multiwords (one per line)")
corpus = load_corpus(url, stopwords, multiwords)

if st.sidebar.checkbox("Show documents"):
	show_documents(corpus)

number_of_topics = st.sidebar.slider("Number of topics", 1, 50, 10)

# Default should be 1. 100 is the value used by Orange (https://orangedatamining.com). We include 
# this option for compatibility with Orange and to examine the impact of this parameter.
number_of_chunks = st.sidebar.slider("Number of chunks", 1, 100, 1)

# The main reason to do this is that the first time a topic model is created, it does not
# seem to be cached properly. Revisit, if this leads to long load times.
if corpus is not None:
	topic_model(corpus, number_of_topics, number_of_chunks)

if st.sidebar.checkbox("Show topics", value=False):
	show_topics(corpus, number_of_topics, number_of_chunks)

if st.sidebar.checkbox("Show document topic matrix", value=False):
	show_document_topic_matrix(corpus, number_of_topics, number_of_chunks)

if st.sidebar.checkbox("Show topic co-occurrences", value=False):
	show_topic_co_occurrences(corpus, number_of_topics, number_of_chunks)

if st.sidebar.checkbox("Show keyword co-occurrences", value=False):
	show_keyword_co_coccurrences(corpus, number_of_topics, number_of_chunks)
