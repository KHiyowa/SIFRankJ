#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unicodedata


def normalize_identity(text):
    return unicodedata.normalize("NFKC", text)


def load_ginza_stopwords(nlp=None):
    try:
        import ginza
    except ImportError:
        ginza = None
    stopwords = set(getattr(ginza, "STOP_WORDS", set())) if ginza is not None else set()
    if not stopwords and nlp is not None:
        stopwords = set(getattr(nlp.Defaults, "stop_words", set()))
    return stopwords | {normalize_identity(word) for word in stopwords}


def set_ginza_split_mode(nlp, split_mode):
    if split_mode is None:
        return
    split_mode = split_mode.upper()
    if split_mode not in {"A", "B", "C"}:
        raise ValueError("split_mode must be one of 'A', 'B', 'C', or None.")
    try:
        import ginza
    except ImportError:
        raise ImportError("split_mode requires GiNZA. Install ginza before using this adapter.")
    ginza.set_split_mode(nlp, split_mode)


class GinzaNLPAdapter:
    """Small StanfordCoreNLP-compatible wrapper for GiNZA/spaCy."""

    POS_MAP = {
        "NOUN": "NN",
        "PROPN": "NNP",
        "ADJ": "JJ",
        "NUM": "NN",
    }

    def __init__(self, model_name="ja_ginza", stopwords=None, disable=None, use_ginza_stopwords=True, split_mode=None):
        try:
            import spacy
        except ImportError:
            raise ImportError("GinzaNLPAdapter requires spaCy and GiNZA. Install them before using this adapter.")

        self.nlp = spacy.load(model_name, disable=disable or [])
        set_ginza_split_mode(self.nlp, split_mode)
        self.split_mode = split_mode.upper() if split_mode is not None else None
        if stopwords is None and use_ginza_stopwords:
            stopwords = load_ginza_stopwords(self.nlp)
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
            if token.text in self.stopwords or normalize_identity(token.text) in self.stopwords or token.is_punct:
                pos = "IN"
            tagged.append((token.text, pos))
        return tagged

    def normalize_token(self, text):
        return normalize_identity(text)

    def normalize_phrase(self, text):
        return normalize_identity(text)

    def close(self):
        return None
