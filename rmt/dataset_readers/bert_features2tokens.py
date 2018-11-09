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
                lazy: bool = False) -> None:
        super().__init__(lazy)
    
    @overrides
    def _read(self, file_path: str):
        tgt_feat_file = file_path + "tgt.jsonl"

        tgt_sentences, tgt_encoded_sentences = self._parse(tgt_feat_file)

        # for i, _ in enumerate(src_sentences):
        #     src = src_sentences[i]
        #     tgt = tgt_sentences[i]
        #     encoded_src = src_encoded_sentences[i]
        #     encoded_tgt = tgt_encoded_sentences[i]
        #     yield self.text_to_instance(encoded_src, encoded_tgt, src, tgt)

        for tgt, encoded_tgt in zip(tgt_sentences, tgt_encoded_sentences):
            yield self.text_to_instance(encoded_tgt, tgt)


    def _parse(self, file):
        sentences = []      
        encoded_sentences = []
        with jsonlines.open(file) as encoded_sentences_dict:   
            for sent in encoded_sentences_dict:
                tokens_dicts = sent["features"]

                sent_tokens = []
                sent_features = []
                for t in tokens_dicts:
                    sent_tokens.append(t["token"])
                    feat = t["layers"][0]["values"]
                    feat = numpy.array([numpy.array(xi) for xi in feat])
                    sent_features.append(feat)

                sentences.append(sent_tokens)
                encoded_sentences.append(numpy.array(sent_features))
        return sentences, encoded_sentences  

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