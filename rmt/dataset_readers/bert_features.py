from allennlp.data import DatasetReader
from allennlp.data.fields import Field, ArrayField, MetadataField
from overrides import overrides
from allennlp.data import Instance
from allennlp.common.file_utils import cached_path
import json
from typing import List, Dict
import numpy

@DatasetReader.register("bert_features")
class BertFeaturesDatasetReader(DatasetReader):
    def __init__(self,
                lazy: bool = False) -> None:
        super().__init__(lazy)
    
    @overrides
    def _read(self, file_path: str):

        with open(cached_path(file_path)) as input_file:
            for i, line in enumerate(input_file):
                src, tgt = line.split("\t")
                src, tgt = json.loads(src), json.loads(tgt)
                src_tokens, src_vectors = self._parse_bert_json(src)
                tgt_tokens, tgt_vectors = self._parse_bert_json(tgt)

                yield self.text_to_instance(src_vectors, tgt_vectors, src_tokens, tgt_tokens)

    def _parse_bert_json(self, jline):
        tokens = []
        vectors = []
        for t_dict in jline['features']:
            tokens.append(t_dict['token'])
            vector = t_dict['layers'][0]['values']
            vector = numpy.array([numpy.array(xi) for xi in vector])
            vectors.append(vector)
        return tokens, numpy.array(vectors)

    @overrides
    def text_to_instance(self,  # type: ignore
                         source_vectors: numpy.ndarray,
                         target_vectors: numpy.ndarray = None,
                         src: str = None,
                         tgt: str = None,
                         ):
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        
        fields["source_vectors"] = ArrayField(array=source_vectors)
        fields["src_strings"] = MetadataField(metadata=src)
        fields["target_vectors"] = ArrayField(array=target_vectors)
        fields["tgt_strings"] = MetadataField(metadata=tgt)

        return Instance(fields)