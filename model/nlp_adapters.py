#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unicodedata


def normalize_identity(text):
    return unicodedata.normalize("NFKC", text)


class GinzaNLPAdapter:
    """Small StanfordCoreNLP-compatible wrapper for GiNZA/spaCy."""

    POS_MAP = {
        "NOUN": "NN",
        "PROPN": "NNP",
        "ADJ": "JJ",
        "NUM": "NN",
    }

    def __init__(self, model_name="ja_ginza", stopwords=None, disable=None):
        try:
            import spacy
        except ImportError:
            raise ImportError("GinzaNLPAdapter requires spaCy and GiNZA. Install them before using this adapter.")

        self.nlp = spacy.load(model_name, disable=disable or [])
        self.stopwords = set(stopwords or [])
        self.phrase_joiner = ""
        self.sentence_delimiters = {"\u3002", "\uff0e", "\uff01", "\uff1f", ".", "!", "?"}
        self._cached_text = None
        self._cached_doc = None

    def _doc(self, text):
        if text != self._cached_text:
            self._cached_text = text
            self._cached_doc = self.nlp(text)
        return self._cached_doc

    def word_tokenize(self, text):
        return [token.text for token in self._doc(text) if not token.is_space]

    def pos_tag(self, text):
        tagged = []
        for token in self._doc(text):
            if token.is_space:
                continue
            pos = self.POS_MAP.get(token.pos_, "IN")
            if token.text in self.stopwords or token.is_punct:
                pos = "IN"
            tagged.append((token.text, pos))
        return tagged

    def normalize_token(self, text):
        return normalize_identity(text)

    def normalize_phrase(self, text):
        return normalize_identity(text)

    def close(self):
        return None
