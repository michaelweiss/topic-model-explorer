import streamlit as st
import numpy as np 
import pandas as pd 
from topics import TopicModel

# model

def load_corpus(url):
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
	show_documents = st.sidebar.checkbox("Show documents", value=True)
	if show_documents:
		show_documents(url)

# application

tm = TopicModel()
app(tm)