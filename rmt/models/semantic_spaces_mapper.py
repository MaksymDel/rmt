from typing import Dict, Optional, List

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import numpy

from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.modules import TimeDistributed, Seq2SeqEncoder
from torch.nn.functional import mse_loss
from allennlp.nn.util import get_mask_from_sequence_lengths

@Model.register("semantic_spaces_mapper")
class SemanticSpacesMapper(Model):
    """
    This class takes as input a sequence of contextualiuzed embeddings in source language
    produced outside (e.g. by BERT) and maps it to the target sequence of contextualiuzed 
    embeddings.

    We also use masking in this class for better mapping.

    Parameters
    ----------

    Returns
    -------

    """
    def __init__(self, vocab: Vocabulary,
                 mapping_layer: Seq2SeqEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        initializer(self)

        self._mapping_layer = mapping_layer

    @overrides
    def forward(self,  # type: ignore
                encoded_src: torch.Tensor,
                encoded_tgt: torch.Tensor,
                src_strings: List[str]=None,
                tgt_strings: List[str]=None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------

        Returns
        -------

        """
        src_mask = self._get_mask_from_token_strings(src_strings)
        tgt_mask = self._get_mask_from_token_strings(tgt_strings) 
        
        estimated_encoded_tgt = self._mapping_layer(encoded_src, src_mask)
        
        loss = mse_loss(estimated_encoded_tgt, encoded_tgt)
        return {"loss": loss}

    @staticmethod
    def _get_mask_from_token_strings(token_strings):
        sequence_lengths = torch.from_numpy(numpy.array([len(l) for l in token_strings]))
        return get_mask_from_sequence_lengths(sequence_lengths, sequence_lengths.max())