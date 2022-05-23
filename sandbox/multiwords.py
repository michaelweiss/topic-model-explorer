import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pke
from nltk.corpus import stopwords
import string
from pyvis.network import Network
import networkx as nx

@st.cache(allow_output_mutation=True)
def extract_keyphrases(text):
	# 1. create a MultipartiteRank extractor.
	extractor = pke.unsupervised.MultipartiteRank()

    # 2. load the content of the document.
	extractor.load_document(input=text)

    # 3. select the longest sequences of nouns and adjectives, that do
	#    not contain punctuation marks or stopwords as candidates.
	pos = {'NOUN', 'ADJ'}
	stoplist = list(string.punctuation)
	stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
	stoplist += stopwords.words('english')
	extractor.candidate_selection(pos=pos, stoplist=stoplist)

    # 4. build the multipartite graph and rank candidates using random walk
	#    alpha controls the weight adjustment mechanism
    #    see TopicRank for threshold and method parameters
	extractor.candidate_weighting(alpha=1.1,
	                              threshold=0.74,
	                              method='average')

	# 5. get the n-highest scored candidates as keyphrases
	keyphrases = extractor.get_n_best(n=50)

	return extractor, keyphrases

def max_weight(topic_graph):
    max_weight = 0
    for edge in topic_graph.edges:
        if topic_graph.edges[edge]['weight'] > max_weight:
            max_weight = topic_graph.edges[edge]['weight']
    return max_weight

def keyphrase_graph(extractor, keyphrases, weight):
    # keyphrases is a list of phrase/phrase weight tuples
    phrases = [phrase[0] for phrase in keyphrases]

    # subgraph of the topic graph with the phrases as nodes
    subgraph = nx.subgraph_view(extractor.graph, 
        lambda x: surface_form(extractor, x) in phrases)

    # convert nodes and edge nodes to their surface forms
    nodes = [surface_form(extractor, node) for node in subgraph.nodes]
    edges = [(surface_form(extractor, edge[0]), surface_form(extractor, edge[1]))
        for edge in subgraph.edges 
            if subgraph.edges[edge[0], edge[1]]['weight'] > weight]

    # create a graph from the nodes and edges
    topic_graph = nx.Graph()
    # topic_graph.add_nodes_from(nodes)
    topic_graph.add_edges_from(edges)

    return topic_graph

def surface_form(extractor, phrase):
    return ' '.join(extractor.candidates[phrase].surface_forms[0]).lower()

def view_keyphrase_graph(topic_graph):
    graph_container = st.empty()
    graph = Network("600px", "100%", notebook=True, heading='')
    graph.from_nx(topic_graph)
    graph.show("topic-graph.html")
    with graph_container.container():
        components.html(open("topic-graph.html", 'r', encoding='utf-8').read(), height=625)

st.title("Multi-word phrases")
st.header("Document")
document = st.text_area("Enter document text", "")
extractor, keyphrases = extract_keyphrases(document)
if st.sidebar.checkbox("Show keyphrases", value=False):
    st.header("Keyphrases")
    for phrase in keyphrases:
        st.markdown(f"{phrase[0]}, {phrase[1]}")
st.header("Keyphrase graph")
norm = max_weight(extractor.graph)
weight = st.sidebar.slider("Min weight", 0.0, 1.0, 0.1, step=0.05)
stopwords = st.sidebar.text_area("Exclude phrases", "")
phrases = [(phrase[0], phrase[1]) for phrase in keyphrases if not phrase[0] in stopwords]
topic_graph = keyphrase_graph(extractor, phrases, weight*norm)
view_keyphrase_graph(topic_graph)

