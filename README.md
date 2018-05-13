# Text classification with Multinomial Naive Bayes implemented in python 3

## Tools
- Install python 3.5+.
- Install [NLTK](http://www.nltk.org/) package for basic NLP processing.

    ```bash
    $ [sudo] pip install nltk
    $ python
    >> import nltk
    >> nltk.download('stopwords')
    ```

- Install [scikit-learn](http://scikit-learn.org/stable/) for fetching 20 newsgroups dataset:

   ```bash
   $ [sudo] pip install scikit-learn
   ```

## Usage
- Run the command below to train data:

    ```bash
    $ python train.py <training size>
    ```

    The classifier will be serialized into `model/classifier.pickle` file.

- Run the command below to test and get accuracy (test size is 5000 by default):

    ```bash
    $ python test.py
    ```
