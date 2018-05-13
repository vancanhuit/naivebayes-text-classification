from utils import helpers


class TestHelpers(object):
    def test_extract_vocabulary(self):
        training_set = [
            ({'a': 3, 'b': 2}, 'L1'),
            ({'a': 1, 'c': 2}, 'L2'),
            ({'b': 3, 'd': 3}, 'L1')
        ]
        vocabulary = helpers.extract_vocabulary(training_set)

        assert type(vocabulary) is set
        assert len(vocabulary) == 4
        assert 'a' in vocabulary
        assert 'b' in vocabulary
        assert 'c' in vocabulary
        assert 'd' in vocabulary

    def test_count_docs_in_label(self):
        training_set = [
            ({'a': 3, 'b': 2}, 'L1'),
            ({'a': 1, 'c': 2}, 'L2'),
            ({'b': 3, 'd': 3}, 'L1')
        ]
        count = helpers.count_docs_in_label(training_set, 'L1')
        assert count == 2

    def test_concencate_text_of_all_docs_in_label(self):
        training_set = [
            ({'a': 3, 'b': 2}, 'L1'),
            ({'a': 1, 'c': 2}, 'L2'),
            ({'b': 3, 'd': 3}, 'L1')
        ]
        label = 'L1'
        docs = helpers.get_all_docs_in_label(
            training_set, label)

        assert type(docs) is list
        assert len(docs) == 2
        assert type(docs[0]) is dict
        assert type(docs[1]) is dict
        assert 'a' in docs[0]
        assert 'd' in docs[1]

    def test_count_tokens_of_term(self):
        docs = [
            {'a': 1, 'b': 2},
            {'a': 2, 'c': 3},
            {'b': 2, 'd': 1}
        ]

        term = 'a'
        count = helpers.count_tokens_of_term(docs, term)
        assert count == 3

    def test_extract_tokens_from_doc(self):
        doc = {'a': 1, 'b': 2, 'c': 3}
        vocabulary = {'a', 'b'}
        tokens = helpers.extract_tokens_from_doc(vocabulary, doc)

        assert type(tokens) is list
        assert len(tokens) == 2
