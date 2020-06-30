##########################################################################
#     Utils
##########################################################################

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd

# Feature creation
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

##########################################################################
#     Code
##########################################################################

def categorical(x):
	if x > 0:
			x = 1
	else:
			x = 0
	return x

def clean(text):
	punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
	for punctuation in punctuation:
		text = text.replace(punctuation, ' ') # Remove Punctuation
	text = text.lower() # Lower Case
	text=text.replace("b ","")
	text=text.replace("b'","")
	text=text.replace('b"',"")
	text = ''.join(c for c in text if not c.isdigit())
	tokenized = word_tokenize(text) # Tokenize
	words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
	stop_words = set(stopwords.words('english')) # Make stopword list
	without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
	lemma=WordNetLemmatizer() # Initiate Lemmatizer
	lemmatized = [lemma.lemmatize(word) for word in without_stopwords] # Lemmatize
	return " ".join(lemmatized)

def polarity(text):
		'''
		positive vs negative sentiment
		Textblob
		'''
		blob = TextBlob(text)
		polarity = blob.sentiment[0]
		return polarity

def subjectivity(text):
		'''
		objectivity vs subjectivity
		objectivity = 0, subjectivity 1
		Textblob
		'''
		blob = TextBlob(text)
		subjectivity = blob.sentiment[1]
		return subjectivity

def compound(text):
		'''
		composure Vader
		'''
		analyzer = SentimentIntensityAnalyzer()
		vs = analyzer.polarity_scores(text)
		return vs['compound']

def negative(text):
		'''
		negativity sentiment Vader
		'''
		analyzer = SentimentIntensityAnalyzer()
		vs = analyzer.polarity_scores(text)
		return vs['neg']

def neutrale(text):
		'''
		neutrality Vader
		'''
		analyzer = SentimentIntensityAnalyzer()
		vs = analyzer.polarity_scores(text)
		return vs['neu']

def positive(text):
		'''
		positivitiy Vader
		'''
		analyzer = SentimentIntensityAnalyzer()
		vs = analyzer.polarity_scores(text)
		return vs['pos']
