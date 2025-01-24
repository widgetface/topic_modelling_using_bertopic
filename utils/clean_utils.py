import re
from typing import List

import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class CleanTextUtils:
    defaultConfig = {
        "lemmatize": True,
        "normalise_text": True,
        "remove_whitespace": True,
        "remove_double_spaces": True,
        "remove_punctuation": True,
        "remove_digits": True,
        "remove_stopwords": True,
        "remove_emoji": False,
        "remove_mentions": False,
        "remove_urls": False,
        "remove_hashtag": False,
        "pos": False,
    }

    def __init__(self, config={}, use_only_config=False, pos_tags=[]):
        self.config = config if use_only_config else self.defaultConfig | config
        self.pos_tags = pos_tags
        self.str = ""
        self.lemmatizer = spacy.load("en_core_web_sm")

    def remove_digits(self):
        no_digit = [i for i in self.str if not i.isdigit()]
        self.str = "".join(no_digit)
        return self

    def remove_emoji(self, string):
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r" ", string)

    def remove_emojis(self):
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        self.str = emoji_pattern.sub(r" ", self.str)
        return self

    def remove_mentions(self):
        # remove user mentions
        self.str = re.sub("@[A-Za-z0-9_]+", " ", self.str)
        return self

    def remove_urls(self):
        # remove URLS
        self.str = re.sub(r"http\S+", " ", self.str)
        return self

    def remove_hashtag(self):
        # remove hashtags
        self.str = re.sub("#[A-Za-z0-9_]+", "", self.str)
        return self

    def remove_emoji(self):
        # remove emoji's
        self.str = self.remove_emojis(self.str)
        return self

    def remove_punctuation(self):
        # remove punctuation
        self.str = re.sub("[^0-9A-Za-z ]", " ", self.str)
        return self

    def remove_double_spaces(self):
        # remove double spaces
        self.str = self.str.replace("  ", " ")

    def remove_whitespace(self):
        self.str = self.str.strip()
        return self

    def remove_stopwords(self):

        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(self.str)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        # with no lower case conversion
        filtered_sentence = []

        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)
        self.str = " ".join(filtered_sentence)
        return self

    def normalise_text(self):
        self.str = self.str.lower()
        return self

    def lemmatize(self):

        doc = self.lemmatizer(self.str)

        # Extract lemmatized tokens
        lemmatized_tokens = [token.lemma_ for token in doc]

        # Join the lemmatized tokens into a sentence
        self.str = " ".join(lemmatized_tokens)
        return self

    def part_of_speech(self):
        doc = self.nlp(self.str)
        self.str = " ".join(
            [token.text for token in doc if token.pos_ in self.pos_tags]
        )
        return self

    @property
    def dataset(self):
        return self.dataset

    def process_dataset(self, dataset):
        config = self.config

        for i, str in enumerate(dataset):
            self.str = str
            if config.get("remove_digits"):
                self.remove_digits()
            if config.get("remove_whitespace"):
                self.remove_whitespace()
            if config.get("normalise_text"):
                self.normalise_text()
            if config.get("remove_double_spaces"):
                self.remove_double_spaces()
            if config.get("remove_punctuation"):
                self.remove_punctuation()
            if config.get("remove_stopwords"):
                self.remove_stopwords()
            if config.get("remove_emoji"):
                self.remove_emoji()
            if config.get("remove_urls"):
                self.remove_urls()
            if config.get("remove_hashtag"):
                self.remove_hashtag()
            if config.get("lemmatize"):
                self.lemmatize()
            dataset[i] = self.str
        if config.get("pos"):
            nlp = spacy.load("en_core_web_sm")
            for i, doc in enumerate(nlp.pipe(dataset)):
                dataset[i] = " ".join(
                    [(ent.text) for ent in doc if ent.pos_ in self.pos_tags]
                )
        return dataset
