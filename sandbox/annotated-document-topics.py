# code fragment to annotate a document with the topics it belongs to

annotation_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
	'#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def annotated_document_topics(corpus, document, topic_model):
	words_and_punctuation = re.findall(r'\w+|\W+', document)
	words = [word for word in corpus.tokenizer.tokenize([corpus.lemmatize(word) for word in corpus.tokenize(document)])]	
	bow = corpus.dictionary.doc2bow(words)
	doc_topics, word_topics, phi_values = topic_model.lda.get_document_topics(bow, per_word_topics=True)
	word2topic = {}
	keywords = []
	for word, topics in word_topics:
		word2topic[corpus.dictionary.id2token[word]] = topics[0]
		keywords.append(corpus.dictionary.id2token[word])
	annotated_words = []
	i = 0
	for word in words_and_punctuation:
		if is_punctuation(word):
			annotated_words.append(word)
		else:
			if words[i] in keywords:
				try:
					annotated_words.append("<span style=\"background-color: {}\">".format(annotation_colors[word2topic[words[i]]]) + 
						word + "</span>")
				except:
					annotated_words.append("<span><u>" + word + "</u></span>")
			else:
				annotated_words.append(word)
			i = i + 1
	return "".join(annotated_words)