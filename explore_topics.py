import streamlit as st
import pandas as pd

import topics

@st.cache(allow_output_mutation=True)
def topic_model():
	print("*** Initialzing the topic model")
	return topics.TopicModel()

@st.cache(allow_output_mutation=True)
def load_corpus(url):
	print("*** Loading corpus: {}".format(url))
	return tm.load_corpus(url)

@st.cache
def lda_model(url, number_of_topics):
	corpus = load_corpus(url)
	with st.spinner("Training the topic model ..."):
		print("*** Training the topic model: {}".format(number_of_topics))
		tm.fit(corpus, number_of_topics)
		print("*** Training completed")

st.sidebar.title("Topic Model Explorer")
tm = topic_model()

#url = st.sidebar.file_uploader("Corpus", type="csv")
#if url is not None:
#	load_corpus(url)
url = "assertions.csv"

show_documents = st.sidebar.checkbox("Show documents", value=True)
if show_documents:
	st.header("Corpus")
	load_corpus(url)
	st.dataframe(tm.corpus.documents)

number_of_topics = st.sidebar.slider("Number of topics", 1, 50, 10)
show_topics = st.sidebar.checkbox("Show topics", value=False)
if show_topics:
	st.header("Topics")
	lda_model(url, number_of_topics)
	st.table(tm.topics(5))