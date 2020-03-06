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

# check the cache if there is already a model for this url, stopwords and number_of_topics
@st.cache(allow_output_mutation=True, persist=True, show_spinner=False)
def lda_model(url, stopwords, number_of_topics):
	with st.spinner("Training the topic model for {} topics ...".format(number_of_topics)):
		print("*** Training the topic model: {}".format(number_of_topics))
		return lda_model_no_cache(url, stopwords, number_of_topics)

def lda_model_no_cache(url, stopwords, number_of_topics):
	corpus = load_corpus(url)
	if use_heuristic_alpha_value:
		return tm.fit(corpus, number_of_topics, alpha="talley", number_of_chunks=number_of_chunks)
	else:
		return tm.fit(corpus, number_of_topics, number_of_chunks=number_of_chunks)

# check the cache if there are already runs for this url, stopwords and number_of_topics
@st.cache(allow_output_mutation=True, persist=True, show_spinner=False)
def lda_model_runs(url, stopwords, number_of_topics, n=5):
	with st.spinner("Creating {} different topic models".format(n)):
		lda_models = [lda_model_no_cache(url, stopwords, number_of_topics) for _ in range(n)]
		return lda_models

def topic_alignment(n):
	lda_models = lda_model_runs(url, stopwords, number_of_topics, n=n)
	topics = pd.DataFrame([[" ".join([tw[0] for tw in lda.lda.show_topic(t, 10)]) for lda in lda_models]
		for t in range(number_of_topics)])
	diff = [lda_models[0].difference(lda_models[i]) for i in range(1, n)]
	matches = pd.DataFrame()
	matches[0] = range(number_of_topics)
	for i in range(1, n):
		_, cols = linear_sum_assignment(diff[i-1])
		matches[i] = cols
	return topics, matches, lda_models, diff

def highlight_topic(x, topic, matches, color="lightgreen"):
	color = "background-color: %s" % (color)
	df = pd.DataFrame('', x.index, x.columns)
	for run in range(len(x.columns)):
		df[run].loc[matches[run][topic]] = color
	return df

def topic_runs(lda_models, topic, matches):
	keywords = pd.DataFrame()
	weights = pd.DataFrame()
	for run in range(len(lda_models)):
		keywords[run] = [tw[0] for tw
			in lda_models[run].lda.show_topic(matches[run][topic], 10)]
		weights[run] = [tw[1] for tw
			in lda_models[run].lda.show_topic(matches[run][topic], 10)]
	return keywords, weights

# done: once we pass weights, use the relative weights to assign colors
# relative weight = weight / lowest weight in top 10
# for all keywords that are repeated across topics, color them blue if all >= 2, 
# green if some >= 2, and blue if all < 2
def highlight_repeated_keywords(keywords, weights):
	df = pd.DataFrame('', keywords.index, keywords.columns)
	num_runs, num_words = len(keywords.columns), len(keywords.index)
	# extract array from data frame
	# we transpose the array so that each row represents one run
	keywords = keywords.values.T
	weights = weights.values.T
	repeated_keywords = []
	for keyword in keywords[0]:
		i = 0
		for run in range(1, num_runs):
			if keyword in keywords[run]:
				i = i + 1
		# print("keyword {} occurs {} times".format(keyword, i))
		if i == num_runs - 1:
			repeated_keywords.append(keyword)
	color = keyword_color(repeated_keywords, num_runs, num_words, keywords, weights)
	for j in range(num_runs):
		for i in range(num_words):
			if keywords[j,i] in repeated_keywords:
				df[j].loc[i] = "background-color: light%s" % (color[keywords[j,i]])
	return df

def keyword_color(repeated_keywords, num_runs, num_words, keywords, weights):
	color = {}
	for keyword in repeated_keywords:
		color[keyword] = None
	for j in range(num_runs):
		for i in range(num_words):
			if keywords[j,i] in repeated_keywords:
				ratio = weights[j,i]/weights[j,num_words-1]
				if ratio >= 2.0:
					if color[keywords[j,i]] is None:
						color[keywords[j,i]] = 'yellow'
					elif color[keywords[j,i]] == 'blue':
						color[keywords[j,i]] = 'green'
				else:
					if color[keywords[j,i]] is None:
						color[keywords[j,i]] = 'blue'
					elif color[keywords[j,i]] == 'yellow':
						color[keywords[j,i]] = 'green'
	return color

def download_link_from_csv(csv, file_name, title="Download"):
	b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
	href = "<a href='data:file/csv;base64,{}' download='{}'>{}</a>".format(b64, file_name, title)
	st.markdown(href, unsafe_allow_html=True)

def download_link_from_html(html, file_name, title="Download"):
	b64 = base64.b64encode(html.encode()).decode()  # some strings <-> bytes conversions necessary here
	href = "<a href='data:file/html;base64,{}' download='{}'>{}</a>".format(b64, file_name, title)
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
use_heuristic_alpha_value = st.sidebar.checkbox("Use heuristic value for alpha", value=False)
number_of_chunks = st.sidebar.slider("Number of chunks", 1, 100, 100)

show_runs = st.sidebar.checkbox("Compare topic model runs", value=False)

if show_runs:
	st.header("Topic Model Runs")
	topic_to_highlight = st.sidebar.selectbox("Highlight topic", range(number_of_topics), 0)
	show_runs_all_topics = st.sidebar.checkbox("Show all topics", value=True)
	if url is None:
		st.markdown("No corpus")
	elif show_runs_all_topics:
		topics, matches, lda_models, diff = topic_alignment(5)
		st.table(topics.style
			.apply(highlight_topic, topic=topic_to_highlight, matches=matches, axis=None))
		download_link_from_csv(topics.to_csv(index=False),
			"tm-{}-runs.csv".format(number_of_topics),
			title="Download topics as CSV")
	else:
		# todo: topic_alignment to return weights as well
		# then pass weights as argument to highlight_repeated_keywords
		topics, matches, lda_models, diff = topic_alignment(5)
		keywords, weights = topic_runs(lda_models, topic=topic_to_highlight, matches=matches)
		st.table(keywords.style
			.apply(highlight_repeated_keywords, weights=weights, axis=None))
		download_link_from_csv(keywords.to_csv(index=False),
			"tm-{}-{}-keywords.csv".format(number_of_topics, topic_to_highlight),
			title="Download keywords as CSV")
		download_link_from_html(keywords.style
			.apply(highlight_repeated_keywords, weights=weights, axis=None)
			.render(), 
			"tm-{}-{}-keywords.html".format(number_of_topics, topic_to_highlight),
			title="Download keywords as HTML (with colors)")
		st.table(weights)
		download_link_from_csv(keywords.to_csv(index=False),
			"tm-{}-{}-weights.csv".format(number_of_topics, topic_to_highlight),
			title="Download weights as CSV")



