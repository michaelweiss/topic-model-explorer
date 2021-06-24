# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import math
import itertools
import base64
import graphviz as graphviz

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
		min_weight = st.sidebar.slider("Minimum weight", 0.0, 0.5, value=0.1)
		min_edges = st.sidebar.slider("Minimum number of edges", 1, 10, value=1)
		st.markdown('''
			We consider topics to co-occur in the same document if the weight of both 
			topics for that document are greater than *minimum weight*. The thickness of
			an edge in the co-occurrance graph indicates how often two topics co-occur
			in a document (at least *minimum edges* times). Each node represents a 
			topic. Node size reflects the total weight of the topic.
		''')
		graph = topic_coocurrence_graph(topic_model(corpus, number_of_topics, number_of_chunks), 
			corpus, number_of_topics, min_weight, min_edges)
		st.graphviz_chart(graph)

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
