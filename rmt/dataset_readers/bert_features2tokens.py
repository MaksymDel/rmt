from allennlp.data import DatasetReader
from allennlp.data.fields import Field, ArrayField, MetadataField, TextField
from overrides import overrides
from allennlp.data import Instance, Token
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common.file_utils import cached_path
import json
import jsonlines
from typing import List, Dict
import numpy

@DatasetReader.register("bert_features2tokens")
class BertFeatures2TokensDatasetReader(DatasetReader):
    def __init__(self,
                lazy: bool = False, for_target_side: bool = True) -> None:
        super().__init__(lazy)
        self._for_target_side = for_target_side

    @overrides
    def _read(self, file_path: str):

        with open(cached_path(file_path)) as input_file:
            for i, line in enumerate(input_file):
                src, tgt = line.split("\t")
                src, tgt = json.loads(src), json.loads(tgt)
                if self._for_target_side:
                    tgt_tokens, tgt_vectors = self._parse_bert_json(tgt)
                    yield self.text_to_instance(tgt_vectors, tgt_tokens)
                else:
                    src_tokens, src_vectors = self._parse_bert_json(src)
                    yield self.text_to_instance(src_vectors, src_tokens)

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
                         target_vectors: numpy.ndarray,
                         tgt: List[str] = None,
                         use_tgt: bool = True):
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        
        fields["target_vectors"] = ArrayField(array=target_vectors)
        if use_tgt:
            fields["target_tokens"] = TextField([Token(t) for t in tgt], {"tokens": SingleIdTokenIndexer()})
        return Instance(fields)