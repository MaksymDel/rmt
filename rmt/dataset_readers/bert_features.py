from allennlp.data import DatasetReader
from allennlp.data.fields import Field, ArrayField, MetadataField
from overrides import overrides
from allennlp.data import Instance
from allennlp.common.file_utils import cached_path
import json
import jsonlines
from typing import List, Dict
import numpy

class BertFeaturesDatasetReader(DatasetReader):
    def __init__(self,
                lazy: bool = False) -> None:
        super().__init__(lazy)
    
    @overrides
    def _read(self, file_path: str):
        src_feat_file = file_path + "src.jsonl"
        tgt_feat_file = file_path + "tgt.jsonl"

        src_sentences, src_encoded_sentences = self._parse(src_feat_file)
        tgt_sentences, tgt_encoded_sentences = self._parse(tgt_feat_file)

        assert len(src_sentences) == len(tgt_sentences)

        # for i, _ in enumerate(src_sentences):
        #     src = src_sentences[i]
        #     tgt = tgt_sentences[i]
        #     encoded_src = src_encoded_sentences[i]
        #     encoded_tgt = tgt_encoded_sentences[i]
        #     yield self.text_to_instance(encoded_src, encoded_tgt, src, tgt)

        for src, tgt, encoded_src, encoded_tgt in zip(src_sentences, tgt_sentences,
                                                      src_encoded_sentences, tgt_encoded_sentences):
            yield self.text_to_instance(encoded_src, encoded_tgt, src, tgt)


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
                         encoded_src: numpy.ndarray, 
                         encoded_tgt: numpy.ndarray,
                         src: str = None,
                         tgt: str = None):
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        
        fields["encoded_src"] = ArrayField(array=encoded_src)
        fields["encoded_tgt"] = ArrayField(array=encoded_tgt)
        fields["src_strings"] = MetadataField(metadata=src)
        fields["tgt_strings"] = MetadataField(metadata=tgt)
        return Instance(fields)