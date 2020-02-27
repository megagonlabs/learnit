import unittest

from learnit.autoconverter.text import FastTextVectorizer


class FastTextVectorizerTestCase(unittest.TestCase):
    def setUp(self):
        self.ft = FastTextVectorizer(min_count=1)
        self.texts = ["I have an apple.",
                      "You hate an apple.",
                      "Apple is red."]

    def test_fit(self):
        X = self.ft.fit_transform(self.texts)
        self.assertEqual(X.shape, (3, 100))
        with self.assertRaises(RuntimeError):
            FastTextVectorizer(min_count=100).fit_transform(self.texts)

    def test_transform(self):
        pass
