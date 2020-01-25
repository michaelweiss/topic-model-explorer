import streamlit as st
import topics

def app():
	st.sidebar.title("Topic Model Explorer")
	url = st.sidebar.text_input("Corpus (URL to a CSV file)", "abstracts.csv")
	load_corpus(url)
	st.table(tm.corpus.documents)

@st.cache(allow_output_mutation=True)
def topic_model():
	return topics.TopicModel()

@st.cache
def load_corpus(url):
	print("*** Loading corpus: {}".format(url))
	tm.load_corpus(url)

tm = topic_model()
app()