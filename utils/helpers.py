def extract_vocabulary(training_set):
    # Each element in training set is a tuple (doc, label)
    # In which, doc is represented as a bag of words
    vocabulary = set()
    for elem in training_set:
        vocabulary.update(elem[0].keys())
    return vocabulary


def count_docs_in_label(training_set, label):
    # Count docs in a given label
    count = 0
    for elem in training_set:
        if elem[1] == label:
            count += 1
    return count


def get_all_docs_in_label(training_set, label):
    # Get all docs belong to a given label
    docs = []
    for elem in training_set:
        if elem[1] == label:
            docs.append(elem[0])
    return docs


def count_tokens_of_term(docs, term):
    # Count tokens of term in docs
    count = 0
    for doc in docs:
        if term in doc.keys():
            count += doc.get(term)
    return count


def extract_tokens_from_doc(vocabulary, doc):
    # Extract tokens from doc
    # In which, doc is represented as a bag of words
    tokens = []
    for token in doc.keys():
        if token in vocabulary:
            tokens.append(token)
    return tokens
