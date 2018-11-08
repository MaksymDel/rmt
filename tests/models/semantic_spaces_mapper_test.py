# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common.testing import ModelTestCase

from rmt import *

class SemanticSpacesAlignmentModelTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model("fixtures/semantic_spaces_mapper/experiment.json",
                          "fixtures/data/")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)