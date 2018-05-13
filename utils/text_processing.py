from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import re


def preprocess_text(text):
    stopwords_set = set(stopwords.words('english'))
    pattern = re.compile(r"[^a-z']")
    replacement = ' '

    processed_text = re.sub(pattern, replacement, text.lower())

    stemmer = PorterStemmer()
    tokens = [
        stemmer.stem(token) for token in processed_text.split() if token
        not in stopwords_set]
    return tokens


def create_bag_of_words(tokens):
    return Counter(tokens)
