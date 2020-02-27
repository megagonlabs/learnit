from __future__ import division

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import FastText


class FastTextVectorizer(BaseEstimator, TransformerMixin):
    """scikit-learn compatible FastText-based Transformer class.

    A model size is proportional to (# of vocabulary) x (# of dimensions).
    The default setting of dimensions is 100. So, this transformer will
    convert textual input into 100 dimensional vectors (= 100 columns).

    Example)
        The model size of FastTextVectorizer fitted on Reuters-21578 corpus
        (14,121 words) is 134MB.

    """

    def __init__(self,
                 tokenizer=None,
                 # Consider single-character words
                 # Original: r"(?u)\b\w\w+\b"
                 token_pattern=r"(?u)\b\w+\b",
                 **kwargs):

        if tokenizer is None:
            self.tokenizer = self._build_tokenizer(token_pattern)
        else:
            self.tokenizer = tokenizer

        self.kwargs = kwargs

    def _build_tokenizer(self,
                         token_pattern=r"(?u)\b\w+\b"):
        """Returns pattern-based tokenizer based on Regex."""
        token_pattern = re.compile(token_pattern)
        return lambda doc: token_pattern.findall(doc)

    def fit(self, X, y=None):
        sentences = list(map(lambda x: self.tokenizer(x), X))
        self.model = FastText(sentences, **self.kwargs)
        return self

    def transform(self, X):
        sentences = list(map(lambda x: self.tokenizer(x), X))
        sent_vec_list = []
        # simple averaging
        for sentence in sentences:
            sent_vec = np.zeros(self.model.vector_size)
            sent_vec_count = 0
            for w in sentence:
                if w in self.model.wv:
                    sent_vec += self.model.wv[w]
                    sent_vec_count += 1
            if sent_vec_count > 0:
                sent_vec /= float(sent_vec_count)
            sent_vec_list.append(sent_vec)
        assert len(sent_vec_list) == len(X)
        return np.array(sent_vec_list)

    def get_feature_names(self):
        assert self.model is not None
        return list(map(lambda x: "fasttext_{}".format(x),
                        range(self.model.vector_size)))