# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np

from topics import TopicModel

# model

@st.cache(allow_output_mutation=True)
def load_corpus(url, stopwords, multiwords):
	return tm.load_corpus(url, stopwords, multiwords)

# view

def show_documents(corpus):
	st.header("Documents")
	if corpus is not None:
		check_for_name_content_columns(corpus.documents)
		st.dataframe(corpus.documents, height=150)
		# download_link_from_csv("\n".join(corpus.stopwords), "stopwords.txt",
		# 	"Download stopwords")
	else:
		st.markdown("Please upload a corpus.")

# view helpers

def check_for_name_content_columns(documents):
	if 'name' not in documents or 'content' not in documents:
		st.markdown('''
		The corpus must have a *name* and a *content* column.
		''')

# controller

tm = TopicModel()

st.sidebar.title("Topic Model Explorer")
st.sidebar.write("Using streamlist {} and gensim {}".format(st.__version__, tm.gensim_version()))

url = st.sidebar.file_uploader("Corpus", type="csv")
stopwords = ""
multiwords = ""
corpus = load_corpus(url, stopwords, multiwords)

if st.sidebar.checkbox("Show documents"):
	show_documents(corpus)

# stopwords = st.sidebar.text_area("Stopwords (one per line)")
# update_stopwords = st.sidebar.button("Update stopwords")

# if update_stopwords:
# 	if url is not None:
# 		corpus = load_corpus(url, "", "")
# 		corpus.update_stopwords(stopwords)
