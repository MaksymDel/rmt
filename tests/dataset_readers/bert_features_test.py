# pylint: disable=invalid-name,no-self-use

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
import numpy
from rmt.dataset_readers.bert_features import BertFeaturesDatasetReader

class TestLatentAlignmentDatasetReader(AllenNlpTestCase):
    def test_reader_can_read(self):
        params = {'lazy': False}
        reader = BertFeaturesDatasetReader.from_params(Params(params))
        dataset = reader.read("fixtures/data/")

        assert len(dataset) == 2
        assert type(dataset[0].fields["source_vectors"].array) == type(dataset[0].fields["target_vectors"].array) == numpy.ndarray
        assert dataset[0].fields["source_vectors"].array.shape[0] == len(dataset[0].fields["src_strings"].metadata)
        assert dataset[0].fields["src_strings"].metadata == ['[CLS]', 'who', 'was', 'jim', 'hen', '##son', '?', '[SEP]'] 
