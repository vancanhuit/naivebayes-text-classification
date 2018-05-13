from utils import text_processing


class TestTextProcessing(object):
    def test_preprocess_text(self):
        text = '''This is a demo text for unit testing. There are some special
        characters such as * / - and + which are all removed after processing
        '''
        tokens = text_processing.preprocess_text(text)
        print(tokens)

    def test_create_bag_of_words(self):
        tokens = ['demo', 'text', 'unit', 'test']
        bow = text_processing.create_bag_of_words(tokens)
        print(bow)
