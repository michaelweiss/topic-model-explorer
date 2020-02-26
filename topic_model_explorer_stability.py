# -*- coding: utf-8 -*-

import streamlit as st
from topics import TopicModel

import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

import base64

@st.cache(allow_output_mutation=True)
def load_corpus(url):
	return tm.load_corpus(url)

@st.cache(allow_output_mutation=True, persist=True, show_spinner=False)
def lda_model(url, stopwords, number_of_topics):
	corpus = load_corpus(url)
	with st.spinner("Training the topic model for {} topics ...".format(number_of_topics)):
		print("*** Training the topic model: {}".format(number_of_topics))
		return lda_model_no_cache(url, stopwords, number_of_topics)

def lda_model_no_cache(url, stopwords, number_of_topics):
	if use_heuristic_alpha_value:
		return tm.fit(corpus, number_of_topics, alpha="talley", number_of_chunks=number_of_chunks)
	else:
		return tm.fit(corpus, number_of_topics, number_of_chunks=number_of_chunks)

@st.cache(allow_output_mutation=True, show_spinner=False)
def lda_model_runs(url, stopwords, number_of_topics, n=4):
	with st.spinner("Creating {} different topic models".format(n)):
		lda_models = [lda_model_no_cache(url, stopwords, number_of_topics) for _ in range(n)]
		return lda_models

def topic_runs(lda_models, topic, matches):
	df = pd.DataFrame()
	for run in range(len(lda_models)):
		df[run] = [tw[0] for tw 
			in lda_models[run].lda.show_topic(matches[run][topic], 10)]
	return df

def topic_alignment(n):
	lda_models = lda_model_runs(url, stopwords, number_of_topics, n=n)
	topic_df = pd.DataFrame([[" ".join([tw[0] for tw in lda.lda.show_topic(t, 10)]) for lda in lda_models]
		for t in range(number_of_topics)])
	diff = [lda_models[0].difference(lda_models[i]) for i in range(1, n)]
	matches = pd.DataFrame()
	matches[0] = range(number_of_topics)
	for i in range(1, n):
		_, cols = linear_sum_assignment(diff[i-1])
		matches[i] = cols
	return topic_df, matches, lda_models, diff

def highlight_topic(x, topic, matches, color="lightgreen"):
	color = "background-color: %s" % (color)
	df = pd.DataFrame('', x.index, x.columns)
	for run in range(len(x.columns)):
		df[run].loc[matches[run][topic]] = color
	return df

def download_link(dataframe, file_name, title="Download"):
	csv = dataframe.to_csv(index=False)
	download_link_from_csv(csv, file_name, title)

def download_link_from_csv(csv, file_name, title="Download"):
	b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
	href = "<a href='data:file/csv;base64,{}' download='{}'>{}</a>".format(b64, file_name, title)
	st.markdown(href, unsafe_allow_html=True)

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
number_of_chunks = st.sidebar.slider("Number of chunks", 1, 100, 100)

show_runs = st.sidebar.checkbox("Compare topic model runs", value=False)

if show_runs:
	st.header("Topic Model Runs")
	topic_to_highlight = st.sidebar.selectbox("Highlight topic", range(number_of_topics), 0)
	show_runs_all_topics = st.sidebar.checkbox("Show all topics", value=True)
	if url is None:
		st.markdown("No corpus")
	elif show_runs_all_topics:
		topic_df, matches, _, _ = topic_alignment(5)
		st.table(topic_df.style
			.apply(highlight_topic, topic=topic_to_highlight, matches=matches, axis=None))
	else:
		topic_df, matches, lda_models, diff = topic_alignment(5)
		topic_runs_df = topic_runs(lda_models, topic=topic_to_highlight, matches=matches)
		st.table(topic_runs_df)


