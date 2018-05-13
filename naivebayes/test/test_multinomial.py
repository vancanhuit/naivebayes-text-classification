from naivebayes import multinomial


class TestMultinomial(object):
    def test_train(self):
        training_set = [
            ({'chinese': 2, 'beijing': 1}, 'yes'),
            ({'chinese': 2, 'shanghai': 1}, 'yes'),
            ({'chinese': 1, 'macao': 1}, 'yes'),
            ({'tokyo': 1, 'japan': 1, 'chinese': 1}, 'no')
        ]

        labels = ['yes', 'no']

        vocabulary, prior, condprob = multinomial.train(training_set, labels)

        print(vocabulary)
        print(prior)
        print(condprob)

    def test_apply(self):
        training_set = [
            ({'chinese': 2, 'beijing': 1}, 'yes'),
            ({'chinese': 2, 'shanghai': 1}, 'yes'),
            ({'chinese': 1, 'macao': 1}, 'yes'),
            ({'tokyo': 1, 'japan': 1, 'chinese': 1}, 'no')
        ]

        labels = ['yes', 'no']

        vocabulary, prior, condprob = multinomial.train(training_set, labels)

        doc = {'tokyo': 1, 'japan': 1, 'chinese': 1}
        label = multinomial.apply(labels, vocabulary, prior, condprob, doc)
        print(label)
