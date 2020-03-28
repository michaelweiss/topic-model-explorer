import streamlit as st
import numpy as np 
import pandas as pd 
from topics import TopicModel

# model

@st.cache(suppress_st_warning=True)
def load_corpus(url):
	st.markdown("Cache miss: load_corpus")
	return tm.load_corpus(url)

# model helpers
# todo: move these into its own module

# view

def show_documents(url):
	st.header("Corpus")
	if url is not None:
		corpus = load_corpus(url)
		check_for_name_content_columns(corpus.documents)
		st.dataframe(corpus.documents)
	else:
		st.markdown("Please upload a corpus.")

def show_stopwords(stopwords):
	st.header("Stopwords")
	if stopwords is not None:
		st.dataframe(stopwords.split('\n'))

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
		show_documents(url)
	stopwords = st.sidebar.text_area("Stopwords (one per line)")
	if st.sidebar.button("Update stopwords"):
		corpus.update_stopwords(stopwords)
		show_stopwords(stopwords)

# application

tm = TopicModel()
app(tm)