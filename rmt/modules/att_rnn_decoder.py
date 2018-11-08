from typing import Dict, List, Tuple

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch


@Seq2SeqEncoder.register("att_rnn2rnn")
class AttentionalRnnDecoder(Seq2SeqEncoder):
    """
    Maps source embeddings to target embeddings ommiting softmax layer. We do not need softmax layer
    because we assume there are "golden vectors" for target words that we are trying to predict
    directly.

    Parameters
    ----------
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : ``int``
        Maximum length of decoded sequences.
    target_embedding_dim : ``int``, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention : ``Attention``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    attention_function: ``SimilarityFunction``, optional (default = None)
        This is if you want to use the legacy implementation of attention. This will be deprecated
        since it consumes more memory than the specialized attention modules.
    scheduled_sampling_ratio : ``float``, optional (default = 0.)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        `Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015 <https://arxiv.org/abs/1506.03099>`_.
    """

    def __init__(self,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 target_embedding_dim: int = None,
                 scheduled_sampling_ratio: float = 0.) -> None:
        super(AttentionalRnnDecoder, self).__init__()
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._encoed_start_symbol = None

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        target_embedding_dim = target_embedding_dim or encoder.get_input_dim()

        # Attention mechanism applied to the encoder output for each step.
        if attention:
            if attention_function:
                raise ConfigurationError("You can only specify an attention module or an "
                                         "attention function, but not both.")
            self._attention = attention
        elif attention_function:
            self._attention = LegacyAttention(attention_function)
        else:
            self._attention = None

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        self._encoder_output_dim = self._encoder.get_output_dim()
        self._decoder_output_dim = self._encoder_output_dim

        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step.
            self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim
        else:
            # Otherwise, the input to the decoder is just the previous target embedding.
            self._decoder_input_dim = target_embedding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder_output_dim, target_embedding_dim)

        # At prediction time, we can use a beam search to find the most likely sequence of target tokens.
        # If the beam_size parameter is not given, we'll just use a greedy search (equivalent to beam_size = 1).
        self._max_decoding_steps = max_decoding_steps

        # Encoded BOS symbol to be set from the outside
        self._encoded_bos_symbol = None
    
    @overrides
    def forward(self,  # type: ignore
                encoded_src: Dict[str, torch.LongTensor],
                src_mask: torch.LongTensor,
                encoded_tgt: Dict[str, torch.LongTensor]=None,
                tgt_mask: torch.LongTensor=None) -> Dict[str, torch.Tensor]:

        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence of embeddings.

        Parameters
        ----------

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        if self._encoded_bos_symbol is None:
            # we set this to bos ymbol from src  since it is always available and the same as 
            # the target's one for many encoders with shared vocabulary like BERT form GoogleAI
            self._set_encoded_bos_symbol(encoded_src[:, 0, :])

        state = self._init_encoded_state(encoded_src, src_mask)
        return self._forward_loop(state, encoded_tgt, tgt_mask)

    def _set_encoded_bos_symbol(self, bos_symbol_embedding: torch.FloatTensor):
        """ Input shape should be (batch_size, embedding_dim) """
        self._encoded_bos_symbol = bos_symbol_embedding

    def _init_encoded_state(self, embedded_input: torch.Tensor, source_mask: torch.LongTensor) -> Dict[str, torch.Tensor]:
        """
        Initialize the encoded state to be passed to the first decoding time step.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        batch_size, _, _ = embedded_input.size()

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)

        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
                encoder_outputs,
                source_mask,
                self._encoder.is_bidirectional())

        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        decoder_hidden = final_encoder_output

        # shape: (batch_size, decoder_output_dim)
        decoder_context = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)

        state = {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
                "decoder_hidden": decoder_hidden,
                "decoder_context": decoder_context
        }

        return state

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      encoded_tgt: torch.FloatTensor=None,
                      tgt_mask: torch.LongTensor=None) -> Dict[str, torch.Tensor]:
        """Make forward pass during training or do greedy search during prediction."""
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]
        encoded_tgt_dim = encoded_tgt.size()[2]

        if encoded_tgt is not None:
            _, target_sequence_length = tgt_mask.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        # Initialize target predictions with the encoded BOS symbol 
        # to feed to the decoder when targets are not available.
        # shape: (batch_size, encoded_tgt_dim)
        # make sure you set self._encoded_bos_symbol in advance) 
        last_predictions_encoded = self._encoded_bos_symbol

        # Append encoded BOS symbol to the output predictions becasuse we want the returnet 
        # sequence of embeddeddings to be in the same format as the input sequence
        step_estimated_tgt_embeddings: List[torch.Tensor] = [last_predictions_encoded.unsqueeze(1)]

        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices_encoded = last_predictions_encoded
            elif encoded_tgt is None:
                # shape: (batch_size,)
                input_choices_encoded = last_predictions_encoded
            else:
                # shape: (batch_size,)
                input_choices_encoded = encoded_tgt[:, timestep, :]

            # shape: (batch_size, encoded_tgt_dim)
            output_projections, state = self._prepare_output_projections(input_choices_encoded, state)

            # list of tensors, shape: (batch_size, 1, encoded_tgt_dim)
            step_estimated_tgt_embeddings.append(output_projections.unsqueeze(1))

            # shape: (batch_size, encoded_tgt_dim)
            last_predictions_encoded = step_estimated_tgt_embeddings

        # shape: (batch_size, num_decoding_steps, num_classes)
        estimated_tgt_embeddings = torch.cat(step_estimated_tgt_embeddings, 1)

        return estimated_tgt_embeddings

    def _prepare_output_projections(self,
                                    embedded_input: torch.FloatTensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()` which is needed for beam search.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = embedded_input

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input,
                (decoder_hidden, decoder_context))

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_hidden)

        return output_projections, state

    def _prepare_attended_input(self,
                                decoder_hidden_state: torch.LongTensor = None,
                                encoder_outputs: torch.LongTensor = None,
                                encoder_outputs_mask: torch.LongTensor = None) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(
                decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input