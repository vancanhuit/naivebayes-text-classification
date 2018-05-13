from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from utils import text_processing
from naivebayes import multinomial
import pickle
import os
import sys


size = int(sys.argv[1])

print('Fetching 20newsgroups dataset with training size is {}...'.format(size))
twenty = fetch_20newsgroups(subset='all')
train_data, _, train_target, _ = train_test_split(
    twenty.data, twenty.target, train_size=size, test_size=5000)
print('Done.')

print('Processing training data...')
training_set = []
labels = twenty.target_names
for index, data in enumerate(train_data):
    tokens = text_processing.preprocess_text(data)
    bow = text_processing.create_bag_of_words(tokens)
    training_set.append((bow, labels[train_target[index]]))
print('Done.')

print('Training...')
vocabulary, prior, condprob = multinomial.train(training_set, labels)
print('Done')

model_file = os.path.join(os.getcwd(), 'model', 'classifier.pickle')
with open(model_file, mode='wb') as f:
    pickle.dump((vocabulary, prior, condprob), f)
