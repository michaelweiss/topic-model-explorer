import streamlit as st
import numpy as np 
import pandas as pd 
from topics import TopicModel

# model

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_corpus(url):
	st.markdown("Cache miss: load_corpus")
	return tm.load_corpus(url)

def update_stopwords(corpus, stopwords):
	if corpus is not None:
		corpus.update_stopwords(stopwords)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def lda_model_runs(corpus, number_of_topics, number_of_chunks, number_of_runs):
	st.markdown("Fitting topic models:")
	progress_bar = st.progress(0)
	lda_models = []
	for run in range(number_of_runs):
		lda_models.append(tm.fit(corpus, number_of_topics, number_of_chunks=number_of_chunks))
		progress_bar.progress(int(100 * (run + 1)/number_of_runs)) 
	return lda_models

# model helpers

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
		lda_models = lda_model_runs(corpus, number_of_topics, number_of_chunks, number_of_runs)
		if st.sidebar.checkbox("Show all topics", value=False):
			st.write(lda_models)
		else:
			st.markdown("Show one topic only")

# view helpers

def check_for_name_content_columns(documents):
	if 'name' not in documents or 'content' not in documents:
		st.markdown('''
		The corpus must have a *name* and a *content* column.
		''')

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