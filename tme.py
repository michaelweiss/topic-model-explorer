# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np

from topics import TopicModel

@st.cache(allow_output_mutation=True)
def load_corpus(url, stopwords, multiwords):
	return tm.load_corpus(url, stopwords, multiwords)

st.sidebar.title("Topic Model Explorer")
tm = TopicModel()

st.sidebar.write("Using streamlist {} and gensim {}".format(st.__version__, tm.gensim_version()))

url = st.sidebar.file_uploader("Corpus", type="csv")
show_documents = st.sidebar.checkbox("Show documents", value=True)

if show_documents:
	st.header("Corpus")
	if url is not None:
		corpus = load_corpus(url, "", "")
		if(corpus is None):
			st.markdown('''
		__Error:__ The corpus must have a *name* and a *content* column.
			''')
		else:
			st.dataframe(corpus.documents, height=150)
			# download_link_from_csv("\n".join(corpus.stopwords), "stopwords.txt",
			# 	"Download stopwords")
	else:
		st.markdown("Please upload a corpus.")
