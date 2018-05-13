def extract_vocabulary(training_set):
    vocabulary = set()
    for elem in training_set:
        vocabulary.update(elem[0].keys())
    return vocabulary


def count_docs_in_label(training_set, label):
    count = 0
    for elem in training_set:
        if elem[1] == label:
            count += 1
    return count


def get_all_docs_in_label(training_set, label):
    docs = []
    for elem in training_set:
        if elem[1] == label:
            docs.append(elem[0])
    return docs


def count_tokens_of_term(docs, term):
    count = 0
    for doc in docs:
        if term in doc.keys():
            count += doc.get(term)
    return count


def extract_tokens_from_doc(vocabulary, doc):
    tokens = []
    for token in doc.keys():
        if token in vocabulary:
            tokens.append(token)
    return tokens
