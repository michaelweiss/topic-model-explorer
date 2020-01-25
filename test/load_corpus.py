import streamlit as st
import topics

def app():
	st.sidebar.title("Topic Model Explorer")
	url = st.sidebar.text_input("Corpus (URL to a CSV file)", "abstracts.csv")
	load_corpus(url)
	st.table(tm.corpus().documents)

@st.cache
def load_corpus(url):
	print("*** Loading corpus: {}".format(url))
	tm = topics.TopicModel()
	tm.load_corpus(url)