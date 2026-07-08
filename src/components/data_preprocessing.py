import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.logger import logging
from src.exception import CustomException


nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


class TextPreprocessor:

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):

        try:
            if not isinstance(text, str):
                return ""

            # Lowercase
            text = text.lower()

            # Remove numbers
            text = re.sub(r"\d+", "", text)

            # Remove punctuation
            text = text.translate(str.maketrans("", "", string.punctuation))

            # Remove extra spaces
            text = " ".join(text.split())

            # Tokenization
            words = text.split()

            # Stopword Removal + Lemmatization
            words = [
                self.lemmatizer.lemmatize(word)
                for word in words
                if word not in self.stop_words
            ]

            return " ".join(words)

        except Exception as e:
            logging.error("Error while preprocessing text.")
            raise CustomException(e)