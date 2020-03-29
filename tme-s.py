import streamlit as st
import numpy as np 
import pandas as pd 
from scipy.optimize import linear_sum_assignment
import time

from topics import TopicModel
from topics import TopicAlignment

# model

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_corpus(url, stopwords):
	st.markdown("Cache miss: load_corpus")
	return tm.load_corpus(url, stopwords)

@st.cache(suppress_st_warning=True)
def find_topic_alignment(corpus, number_of_topics, number_of_chunks, number_of_runs):
	status = st.markdown("Fitting topic models:")
	progress_bar = st.progress(0)
	def progress_update(run):
		progress_bar.progress(int(100 * (run + 1)/number_of_runs))
	alignment = TopicAlignment(tm, corpus, number_of_topics, number_of_chunks, number_of_runs)
	alignment.fit(progress_update)
	hide_status_indicators(status, progress_bar)
	return alignment

# model helpers

# view

def show_documents(corpus):
	st.header("Documents")
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
		selected_topic = st.sidebar.selectbox("Select topic to highlight", range(number_of_topics), 0)
		alignment = find_topic_alignment(corpus, number_of_topics, number_of_chunks, number_of_runs)
		if st.sidebar.checkbox("Show all topics", value=False):
			"""
			Highlighting shows which topics correspond across the different runs.
			The alignment is calculated based on minimizing the sum of the differences between topics.
			"""
			st.table(alignment.topics.style
				.apply(highlight_topic, topic=selected_topic, matches=alignment.matches, axis=None))
		else:
			"""
			Show a single topic only.
			"""

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
	stopwords = st.sidebar.text_area("Stopwords (one per line)")
	corpus = load_corpus(url, stopwords)
	if st.sidebar.checkbox("Show documents"):
		show_documents(corpus)
	number_of_topics = st.sidebar.slider("Number of topics", 1, 50, 10)
	number_of_chunks = st.sidebar.slider("Number of chunks", 1, 100, 100)
	number_of_runs = st.sidebar.slider("Number of runs", 1, 10, 4)
	if st.sidebar.checkbox("Show topic model runs", value=False):
		show_topic_model_runs(corpus, number_of_topics, number_of_chunks, number_of_runs)

# application

tm = TopicModel()
app(tm)