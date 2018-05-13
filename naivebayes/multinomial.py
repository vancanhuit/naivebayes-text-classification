from utils import helpers
import math


def train(training_set, labels):
    vocabulary = helpers.extract_vocabulary(training_set)
    condprob = {}
    prior = {}

    num_docs = len(training_set)
    for label in labels:
        num_docs_in_label = helpers.count_docs_in_label(training_set, label)
        prior[label] = num_docs_in_label / num_docs

        docs_in_label = helpers.get_all_docs_in_label(
            training_set, label)

        temp = {}
        for term in vocabulary:
            temp[term] = {}
            temp[term][label] = helpers.count_tokens_of_term(
                docs_in_label, term)

        denominator = sum((temp[term][label] + 1) for term in vocabulary)
        for term in vocabulary:
            condprob.setdefault(term, {})
            condprob[term][label] = (temp[term][label] + 1) / denominator

    return vocabulary, prior, condprob


def apply(labels, vocabulary, prior, condprob, doc):
    tokens = helpers.extract_tokens_from_doc(vocabulary, doc)
    scores = [0] * len(labels)

    for index, label in enumerate(labels):
        scores[index] = math.log(prior[label])
        for token in tokens:
            scores[index] += math.log(condprob[token][label])

    return labels[scores.index(max(scores))]
