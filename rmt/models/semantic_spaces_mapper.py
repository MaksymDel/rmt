from typing import Dict, Optional, List

from overrides import overrides
import torch
import numpy

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from torch.nn.functional import mse_loss
from allennlp.nn.util import get_mask_from_sequence_lengths, masked_mean
from allennlp.models.archival import load_archive, Archive
from allennlp.common.checks import ConfigurationError

from rmt.modules.att_rnn_decoder import AttentionalRnnDecoder
from rmt.models.semantic_space_decoder import SemanticSpaceDecoder

@Model.register("semantic_spaces_mapper")
class SemanticSpacesMapper(Model):
    """
    This class takes as input a sequence of contextualized embeddings in source language
    produced outside (e.g. by BERT) and maps it to the target sequence of contextualized 
    embeddings.

    Basically it follows functionality of usual Seq2Seq decoder.

    We also use masking in this class for better mapping.

    Parameters
    ----------

    Returns
    -------

    """
    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,
                 mapping_layer: Seq2SeqEncoder, # AttentionalRnnDecoder actually
                 path_to_generator: str = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        initializer(self)

        assert encoder.get_output_dim() == mapping_layer.get_input_dim()

        self._encoder = encoder # can be bypassed (use BypassEncoder) since input is already encoded with e.g. BERT
        self._mapping_layer = mapping_layer

        self._semantic_space_decoder = None
        # You should pass it as overrides argument at test time when using `allennlp predict` command
        if path_to_generator is not None:
            self._path_to_generator = path_to_generator

    @overrides
    def forward(self,  # type: ignore
                source_vectors: torch.Tensor,
                src_strings: List[str],
                target_vectors: torch.Tensor = None,
                tgt_strings: List[str] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------

        Returns
        -------

        """
        src_mask = self._get_mask_from_token_strings(src_strings)
        # tgt_mask = self._get_mask_from_token_strings(tgt_strings) 

        # prepare source vectors for decoding
        state = self._init_encoded_state(source_vectors, src_mask)

        # generate target embeddings
        estimated_target_vectors = self._mapping_layer(state, target_vectors)

        # compute MSE loss wrt golden target embeddings
        assert target_vectors.size() == estimated_target_vectors.size()
        loss = mse_loss(estimated_target_vectors, target_vectors)
        return {"loss": loss, "estimated_target_vectors": estimated_target_vectors}

    def maybe_init_with_semantic_space_decoder(self):
        if self._semantic_space_decoder is None:
            if self._path_to_generator is None:
                raise ConfigurationError("Your config should contain `path_to_generator` parameter to infer")         
            self._semantic_space_decoder = load_archive(self._path_to_generator, self._get_prediction_device()).model

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.
        
        This method passes predicted target vectors to target semantic space decoder model
        which generate natural langauge from that.
        """
        self.maybe_init_with_semantic_space_decoder()
        
        output_dict = self._semantic_space_decoder(output_dict["estimated_target_vectors"])
        output_dict = self._semantic_space_decoder.decode(output_dict)

        return {"predicted_tokens": output_dict["predicted_tokens"]}

    def _init_encoded_state(self, embedded_input: torch.Tensor, source_mask: torch.LongTensor) -> Dict[str, torch.Tensor]:
        """
        Initialize the encoded state to be passed to the first decoding time step.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        batch_size, _, _ = embedded_input.size()

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)

        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        decoder_hidden = self._get_summary_of_encoder_outputs(encoder_outputs, source_mask)
        
        # shape: (batch_size, decoder_output_dim)
        decoder_context = encoder_outputs.new_zeros(batch_size, self._mapping_layer.get_output_dim())

        state = {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
                "decoder_hidden": decoder_hidden,
                "decoder_context": decoder_context,
                "encoded_bos_symbol": embedded_input[:, 0, :]
        }

        return state

    def _get_summary_of_encoder_outputs(self, encoder_outputs, source_mask):
        # This returns last final encoder output in case of RNN encoders,
        # and mean of the outputs in case of other encoders
        if type(self._encoder) == PytorchSeq2SeqWrapper:
            summary = util.get_final_encoder_states(
                                                encoder_outputs,
                                                source_mask,
                                                self._encoder.is_bidirectional())
        else:
            summary = masked_mean(encoder_outputs, source_mask.unsqueeze(-1).to(encoder_outputs.device), dim=1, keepdim=False)
        return summary

    @staticmethod
    def _get_mask_from_token_strings(token_strings):
        sequence_lengths = torch.from_numpy(numpy.array([len(l) for l in token_strings]))
        return get_mask_from_sequence_lengths(sequence_lengths, sequence_lengths.max())
