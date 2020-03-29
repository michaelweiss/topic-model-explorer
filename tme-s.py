import streamlit as st
import numpy as np 
import pandas as pd 
from scipy.optimize import linear_sum_assignment
import time

from topics import TopicModel

# model

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_corpus(url):
	st.markdown("Cache miss: load_corpus")
	return tm.load_corpus(url)

def update_stopwords(corpus, stopwords):
	if corpus is not None:
		corpus.update_stopwords(stopwords)

@st.cache(suppress_st_warning=True)
def lda_model_runs(corpus, number_of_topics, number_of_chunks, number_of_runs):
	status = st.markdown("Fitting topic models:")
	progress_bar = st.progress(0)
	lda_models = []
	for run in range(number_of_runs):
		lda_models.append(tm.fit(corpus, number_of_topics, number_of_chunks=number_of_chunks))
		# lda_models.append([run, number_of_topics, number_of_chunks])
		# time.sleep(1)
		progress_bar.progress(int(100 * (run + 1)/number_of_runs))
	hide_status_indicators(status, progress_bar)
	return lda_models

# model helpers

def topic_alignment(lda_models, number_of_topics):
	# extract the topic words for each topic in all topic models
	topics = pd.DataFrame([[" ".join([tw[0] for tw in lda.lda.show_topic(t, 10)]) 
		for lda in lda_models] for t in range(number_of_topics)])
	# compute the average Jaccard distance between the topic models
	diffs = [lda_models[0].difference(lda_models[i]) 
		for i in range(1, len(lda_models))]
	matches = pd.DataFrame()
	# first column are the topics of the first topic model
	matches[0] = range(number_of_topics)
	# try to fit topics between the first and each of the remaining topic models
	# the Hungarian assignment method uses the degree of match between topics (diffs)
	# and minimizes the total misalignment between topics
	for i in range(1, len(lda_models)):
		_, cols = linear_sum_assignment(diffs[i-1])
		# each column contains the topics that match the topics of the first topic
		matches[i] = cols
	return topics, matches, diffs

# view

def show_documents(corpus):
	st.header("Corpus")
	if corpus is not None:
		check_for_name_content_columns(corpus.documents)
		st.dataframe(corpus.documents)
	else:
		st.markdown("Please upload a corpus.")

def show_topic_model_runs(corpus, number_of_topics, number_of_chunks, number_of_runs, show_all=True):
	st.header("Topic model runs")
	if corpus is None:
		st.markdown("Please upload a corpus first")
	else:
		selected_topic = st.sidebar.selectbox("Highlight topic", range(number_of_topics), 0)
		lda_models = lda_model_runs(corpus, number_of_topics, number_of_chunks, number_of_runs)
		# topics, matches, diffs = topic_alignment(lda_models, number_of_topics)
		if st.sidebar.checkbox("Show all topics", value=False):
			st.write(lda_models)
		# 	st.table(topics.style
		# 		.apply(highlight_topic, topic=selected_topic, matches=matches, axis=None))
		else:
		 	st.markdown("Show one topic only")

# view helpers

def check_for_name_content_columns(documents):
	if 'name' not in documents or 'content' not in documents:
		st.markdown('''
		The corpus must have a *name* and a *content* column.
		''')

def highlight_topic(x, topic, matches, color="lightgreen"):
	color = "background-color: %s" % (color)
	df = pd.DataFrame('', x.index, x.columns)
	for run in range(len(x.columns)):
		df[run].loc[matches[run][topic]] = color
	return df

def hide_status_indicators(*indicators):
	time.sleep(1)
	for indicator in indicators:
		indicator.empty()

# controller

def app(tm):
	st.sidebar.title("Topic Model Explorer")
	url = st.sidebar.file_uploader("Corpus", type="csv")
	corpus = load_corpus(url)
	if st.sidebar.checkbox("Show documents"):
		show_documents(corpus)
	stopwords = st.sidebar.text_area("Stopwords (one per line)")
	if st.sidebar.button("Update stopwords"):
		update_stopwords(corpus, stopwords)
	number_of_topics = st.sidebar.slider("Number of topics", 1, 50, 10)
	number_of_chunks = st.sidebar.slider("Number of chunks", 1, 100, 100)
	number_of_runs = st.sidebar.slider("Number of runs", 1, 10, 4)
	if st.sidebar.checkbox("Show topic model runs", value=False):
		show_topic_model_runs(corpus, number_of_topics, number_of_chunks, number_of_runs)

# application

tm = TopicModel()
app(tm)