def keyword_coocurrence_graph(selected_topic, min_edges, cut_off, source, topic_depth):
	corpus = load_corpus(url)
	dtm = document_topics_matrix()
	top_documents = sort_by_topic(dtm, selected_topic, cut_off)
	print("*** top documents: {}".format(top_documents))
	top_topic_keywords = topic_words(selected_topic, topic_depth)
	def include_word(word):
		if source == "Exclude stopwords":
			return word not in corpus.stopwords_en
		else:
			return word in top_topic_keywords
	documents = corpus.documents['content'][top_documents]
	index = {}
	reverse_index = {}
	next_index = 0
	sentence_words = []
	for document in documents:
		for sentence in document.split(". "):
			sentence = re.sub(r'[^A-Za-z0-9]+', ' ', sentence)
			words = [word for word in sentence.lower().split(" ") 
					if include_word(word)]
			# words = [word for word in sentence.lower().split(" ") 
			# 	if word not in corpus.stopwords_en]
			words = set(words)
			for word in words:
				if word not in index:
					index[word] = next_index
					reverse_index[next_index] = word
					next_index = next_index + 1
			sentence_words.append(words)
	edge = np.zeros((len(index), len(index)))
	for words in sentence_words:
		for wi, wj in list(itertools.combinations(words, 2)):
			if wi < wj:
				edge[index[wi], index[wj]] = edge[index[wi], index[wj]] + 1
			else:
				edge[index[wj], index[wi]] = edge[index[wj], index[wi]] + 1
	graph = graphviz.Graph(format='png')
	graph.attr('node', shape='plaintext')
	nodes = []
	for i in range(len(index)):
		for j in range(len(index)):
			if edge[i, j] >= min_edges:
				if i not in nodes:
					nodes.append(i)
				if j not in nodes:
					nodes.append(j)
				graph.edge(reverse_index[i], reverse_index[j], 
					penwidth="{}".format(math.sqrt(edge[i, j])))
	for i in nodes:
		graph.node(reverse_index[i])
	# todo: create dataframe with top documents and their name/content
	# pd.DataFrame(dtm).iloc[top_documents]
	return graph, [reverse_index[node] for node in nodes], top_documents