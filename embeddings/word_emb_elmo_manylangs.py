#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from elmoformanylangs import Embedder


class WordEmbeddings():
    """ELMoForManyLangs embedding wrapper.

    Expects a model directory containing config.json, encoder.pkl,
    token_embedder.pkl, word.dic, and char.dic.
    """

    def __init__(self, model_dir="../auxiliary_data/ja.model", batch_size=64, cuda_device=0):
        self.cuda_device = cuda_device
        self.elmo = Embedder(model_dir, batch_size=batch_size)

    def get_tokenized_words_embeddings(self, sents_tokened):
        embeddings = []
        for emb in self.elmo.sents2elmo(sents_tokened, output_layer=-2):
            if emb.ndim == 2:
                emb = np.stack([emb, emb, emb], axis=0)
            embeddings.append(emb)

        batch_size = len(embeddings)
        max_len = max((emb.shape[1] for emb in embeddings), default=0)
        emb_dim = embeddings[0].shape[2] if embeddings else 0

        elmo_embedding = torch.zeros((batch_size, 3, max_len, emb_dim), dtype=torch.float32)
        elmo_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for i, emb in enumerate(embeddings):
            length = emb.shape[1]
            elmo_embedding[i, :, :length, :] = torch.from_numpy(emb).float()
            elmo_mask[i, :length] = True

        return elmo_embedding, elmo_mask
