import streamlit as st
import topics

@st.cache_resource
def topic_model():
	print("*** Initialzing the topic model")
	return topics.TopicModel()

@st.cache_data
def load_corpus(url):
	print("*** Loading corpus: {}".format(url))
	return tm.load_corpus(url)

tm = topic_model()
st.sidebar.title("Topic Model Explorer")

url = st.sidebar.file_uploader("Corpus", type=["csv"])
if url is not None:
	corpus = load_corpus(url)
	st.table(tm.corpus.documents)

