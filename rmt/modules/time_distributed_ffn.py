from typing import List

from overrides import overrides
import torch

from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.activations import Activation
from allennlp.modules import TimeDistributed

@Seq2SeqEncoder.register("time_distributed_ff")
class TimeDistributedFeedForwardEncoder(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    Parameters
    ----------
    """
    def __init__(self, feed_forward: FeedForward) -> None:
        super(TimeDistributedFeedForwardEncoder, self).__init__()
        self._input_dim = feed_forward.get_input_dim()
        self._output_dim = feed_forward.get_output_dim()
        self._time_distributed_fnn = TimeDistributed(feed_forward)
    
    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None): # pylint: disable=arguments-differ
        return self._time_distributed_fnn(inputs)
            
    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim
