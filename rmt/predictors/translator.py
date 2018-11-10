from overrides import overrides
import json

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('translator')
class Translator(Predictor):
    """
    Predictor for the :class:`~allennlp.models.encoder_decoder.simple_seq2seq` model.
    """

    # def predict(self, source: str) -> JsonDict:
    #     return self.predict_json({"source" : source})

    # @overrides
    # def _json_to_instance(self, json_dict: JsonDict) -> Instance:
    #     """
    #     Expects JSON that looks like ``{"source": "..."}``.
    #     """
    #     source = json_dict["source"]
    #     return self._dataset_reader.text_to_instance(source)

    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return json.dumps(" ".join(outputs["predicted_tokens"])).replace(" ##", "")[1:-1]  + "\n"  # here we remove first and last symbols which are quotes