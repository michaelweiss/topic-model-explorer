import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

words = ["resource", "resources",
	"company", "companies",
	"run", "ran",
	"like", "likes"]

for w in words:
	print("{} = {}".format(w, lemmatizer.lemmatize(w)))

from nltk.tokenize import MWETokenizer

tokenizer = MWETokenizer()
tokenizer.add_mwe(('open', 'source'))
words = tokenizer.tokenize('The governance of open source projects'.split())

print(" ".join(words))