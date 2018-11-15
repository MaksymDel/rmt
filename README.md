# Repository for Research in Regressional NMT by Mapping Pretrained Semantic Spaces [Closed]


The idea was to encode 2 languages with BERT. 
Then learn 2 decoders for these languages (from bert embeddings).
Lastly we could learn a mapping between bert feature spaces for 2 languages, and do translation this way.
Mapping could be also learned in unsupervised way (e.g. with GANs) since embeddings are differentiable.

Result: it was too hard task for decoder to decode from BERT embeddings into tokens (10 BLEU after 30 epochs of training.)
Probably need to use more data/stronger model. 

Alternatively one can learn to autoencoders and map their codes. 
Very similar (same?) experiment is discussed here however: https://arxiv.org/abs/1706.04223

```
0) create new conda env
1) pip install -r requrements.txt
2) git clone https://github.com/google-research/bert
3) pip install -r bert/requrements.txt
4) cd bert
5) wget https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip
6) unzip multilingual_L-12_H-768_A-12.zip
7) cd .. 
8) bash ../scripts/extract_features_bert.sh
9) wget https://download.pytorch.org/models/translate/iwslt14/data.tar.gz
10) allennlp train fixtures/semantic_space_decoder/experiment.json -s fixtures/semantic_space_decoder/serialization --include-package rmt -f
11) allennlp train fixtures/semantic_spaces_mapper/experiment.json -s fixtures/semantic_spaces_mapper/serialization --include-package rmt -f
12) allennlp predict fixtures/semantic_spaces_mapper/serialization/model.tar.gz fixtures/data/ --include-package rmt --predictor translator --use-dataset-reader -o "{"model":{"path_to_generator": 'fixtures/semantic_space_decoder/serialization/model.tar.gz'}}"


a) allennlp train training_config/decoders/experiment.json -s output/decoders/serialization --include-package rmt -f
b) allennlp train training_config/mappers/experiment.json -s output/mappers/serialization --include-package rmt -f
c) allennlp predict output/mappers/serialization/model.tar.gz ../rmt-data/iwslt14-de-en/test/ --include-package rmt --predictor translator --use-dataset-reader -o "{"model":{"path_to_generator": 'output/decoders/serialization/model.tar.gz'}}"
```