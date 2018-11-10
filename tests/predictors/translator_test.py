# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive, Archive
from allennlp.predictors import Predictor

import rmt

class TestTranslator(AllenNlpTestCase):
    def test_loads(self):
        # archive: Archive = load_archive('fixtures/semantic_space_decoder/serialization/model.tar.gz')        
        # model = archive.model 
        # archive = load_archive('fixtures/semantic_spaces_mapper/serialization/model.tar.gz', 
        #                     overrides="{'model': {'path_to_generator': 'fixtures/semantic_space_decoder/serialization/model.tar.gz'}}")
        
        archive = load_archive('fixtures/semantic_spaces_mapper/serialization/model.tar.gz', overrides="{'model': {'path_to_generator': 'fixtures/semantic_space_decoder/serialization/model.tar.gz'}}")
        a = 2+2
        #predictor = Predictor.from_archive(archive, 'translator')