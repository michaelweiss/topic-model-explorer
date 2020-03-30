import streamlit as st
import numpy as np 
import pandas as pd 
from scipy.optimize import linear_sum_assignment
import time
import base64

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
		download_link_from_csv("\n".join(corpus.stopwords), "stopwords.txt",
			"Download stopwords")
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
			This table gives an overview of the topics in each run.

			Highlighting shows which topics correspond across the different runs.
			The alignment is calculated based on minimizing the sum of the differences between topics.
			
			Note: The difference between two topics is computed as the average Jaccard distance, which measures
			the overlap between consecutive subsets of the first N keywords in both topics.
			"""
			st.table(alignment.topics.style
				.apply(highlight_topic, topic=selected_topic, matches=alignment.matches, axis=None))
			download_link_from_csv(alignment.topics.to_csv(index=False), 
				"tm-{}-runs.csv".format(number_of_topics), 
				"Download topic model runs")
		else:
			"""
			This table shows the alignment across runs for a single topic.

			Keywords that are repeated across most topics are highlighted. The threshold is set
			to 75% of the topics, so for 4 runs, keywords will be highlighted if they are repeated 
			across 3 or more runs.

			Colors are assigned based on relative keyword weight (ratio of keyword weight 
			and the lowest keyword weight for the top-10 keywords) for a given run. Keywords 
			are colored *yellow* if the ratio is >= 2 across all runs, *green* if it is >= 2 for 
			some of the runs, and *blue* if it is < 2 for all runs.
			"""
			st.table(alignment.keywords[selected_topic].style
				.apply(highlight_repeated_keywords, weights=alignment.weights[selected_topic], 
					min_runs=int(number_of_runs * 0.75), axis=None))
			download_link_from_csv(alignment.keywords[selected_topic].to_csv(index=False), 
				"tm-{}-{}-keywords.csv".format(number_of_topics, selected_topic), 
				"Download keywords")
			download_link_from_html(alignment.keywords[selected_topic].style
				.apply(highlight_repeated_keywords, weights=alignment.weights[selected_topic], 
					min_runs=int(number_of_runs * 0.75), axis=None).render(), 
				"tm-{}-{}-keywords.html".format(number_of_topics, selected_topic),
				title="Download keywords (with colors)")

# view helpers

def check_for_name_content_columns(documents):
	if 'name' not in documents or 'content' not in documents:
		st.markdown('''
		The corpus must have a *name* and a *content* column.
		''')

def hide_status_indicators(*indicators):
	time.sleep(1)
	for indicator in indicators:
		indicator.empty()

def highlight_topic(x, topic, matches, color="lightgreen"):
	color = "background-color: %s" % (color)
	df = pd.DataFrame('', x.index, x.columns)
	for run in range(len(x.columns)):
		df[run].loc[matches[run][topic]] = color
	return df

def highlight_repeated_keywords(keywords, weights, min_runs):
	df = pd.DataFrame('', keywords.index, keywords.columns)
	num_runs, num_words = len(keywords.columns), len(keywords.index)
	# extract array from data frame
	# we transpose the array so that each row represents one run
	keywords = keywords.values.T
	weights = weights.values.T
	repeated_keywords = []
	for keyword in keywords[0]:
		# todo: change index, i is used to represent the run elsewhere
		i = 0
		for run in range(1, num_runs):
			if keyword in keywords[run]:
				i = i + 1
		# add keyword to repeated_keywords if it occurs in each run
		# can modify this to some percentage of the runs		
		if i >= min_runs - 1:
			repeated_keywords.append(keyword)
	color = keyword_color(repeated_keywords, num_runs, num_words, keywords, weights)
	for j in range(num_runs):
		for i in range(num_words):
			if keywords[j,i] in repeated_keywords:
				df[j].loc[i] = "background-color: light%s" % (color[keywords[j,i]])
	return df

# for all keywords that are repeated across topics, color them blue if all >= 2, 
# green if some >= 2, and blue if all < 2
def keyword_color(repeated_keywords, num_runs, num_words, keywords, weights):
	color = {}
	for keyword in repeated_keywords:
		color[keyword] = None
	for i in range(num_runs):
		for j in range(num_words):
			if keywords[i,j] in repeated_keywords:
				ratio = weights[i,j]/weights[i,num_words-1]
				if ratio >= 2.0:
					if color[keywords[i,j]] is None:
						color[keywords[i,j]] = 'yellow'
					elif color[keywords[i,j]] == 'blue':
						color[keywords[i,j]] = 'green'
				else:
					if color[keywords[i,j]] is None:
						color[keywords[i,j]] = 'blue'
					elif color[keywords[i,j]] == 'yellow':
						color[keywords[i,j]] = 'green'
	return color

def download_link_from_csv(csv, file_name, title="Download"):
	b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
	href = "<a href='data:file/csv;base64,{}' download='{}'>{}</a>".format(b64, file_name, title)
	st.markdown(href, unsafe_allow_html=True)

def download_link_from_html(html, file_name, title="Download"):
	b64 = base64.b64encode(html.encode()).decode()  # some strings <-> bytes conversions necessary here
	href = "<a href='data:file/html;base64,{}' download='{}'>{}</a>".format(b64, file_name, title)
	st.markdown(href, unsafe_allow_html=True)

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