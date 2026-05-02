#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19
import torch

try:
    from allennlp.commands.elmo import ElmoEmbedder
except ImportError:
    from allennlp.modules.elmo import _ElmoBiLm, batch_to_ids
    from allennlp.nn.util import remove_sentence_boundaries

    class ElmoEmbedder():
        """Compatibility wrapper for AllenNLP 2.x ELMo."""

        def __init__(self, options_file, weight_file, cuda_device=0):
            self.cuda_device = cuda_device
            self.elmo = _ElmoBiLm(options_file, weight_file)
            if self.cuda_device > -1:
                self.elmo = self.elmo.cuda(self.cuda_device)
            self.elmo.eval()

        def batch_to_embeddings(self, sents_tokened):
            character_ids = batch_to_ids(sents_tokened)
            if self.cuda_device > -1:
                character_ids = character_ids.cuda(self.cuda_device)
            with torch.no_grad():
                bilm_output = self.elmo(character_ids)
                activations = []
                for activation in bilm_output["activations"]:
                    activation, _ = remove_sentence_boundaries(activation, bilm_output["mask"])
                    activations.append(activation)
                elmo_embedding = torch.stack(activations, dim=1)
                elmo_mask = character_ids.sum(dim=-1) > 0
            return elmo_embedding, elmo_mask

class WordEmbeddings():
    """
        ELMo
        https://allennlp.org/elmo

    """

    def __init__(self,
                 options_file="../auxiliary_data/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                 weight_file="../auxiliary_data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5", cuda_device=0):
        self.cuda_device=cuda_device
        self.elmo = ElmoEmbedder(options_file, weight_file,cuda_device=self.cuda_device)

    def get_tokenized_words_embeddings(self, sents_tokened):
        """
        @see EmbeddingDistributor
        :param tokenized_sents: list of tokenized words string (sentences/phrases)
        :return: ndarray with shape (len(sents), dimension of embeddings)
        """

        elmo_embedding, elmo_mask = self.elmo.batch_to_embeddings(sents_tokened)
        if(self.cuda_device>-1):
            return elmo_embedding.cpu(), elmo_mask.cpu()
        else:
            return elmo_embedding, elmo_mask

