import os
import pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from naivebayes import multinomial
from utils import text_processing


twenty = fetch_20newsgroups(subset='all')
_, test_data, _, test_target = train_test_split(
    twenty.data, twenty.target, test_size=5000)


model_file = os.path.join(os.getcwd(), 'model', 'classifier.pickle')
with open(model_file, mode='rb') as f:
    vocabulary, prior, condprob = pickle.load(f)

labels = twenty.target_names
count = 0
print('Testing...')
for index, data in enumerate(test_data):
    tokens = text_processing.preprocess_text(data)
    bow = text_processing.create_bag_of_words(tokens)
    label = multinomial.apply(labels, vocabulary, prior, condprob, bow)
    if label == labels[test_target[index]]:
        count += 1

accuracy = count / len(test_data) * 100
print('Accuracy: {0:.2f}%'.format(accuracy))
