export PROJ_ROOT=/home/maksym/research/rmt
export BERT_DIR=$PROJ_ROOT/bert
export BERT_MODEL_DIR=$BERT_DIR/multilingual_L-12_H-768_A-12

export LAYERS=-1
export MAX_SEQ_LEN=128
export BATCH_SIZE=40

#export BERT_INPUT_FILE=$PROJ_ROOT/fixtures/src.txt
#export BERT_OUTPUT_FILE=$PROJ_ROOT/fixtures/src.jsonl

# DATASET
export DATASET_FOLDER_NAME=iwslt14-de-en

# SOURCE
export BERT_INPUT_FILE=$PROJ_ROOT/data/$DATASET_FOLDER_NAME/train/src.txt
export BERT_OUTPUT_FILE=$PROJ_ROOT/data/$DATASET_FOLDER_NAME/train/src.jsonl

python $BERT_DIR/extract_features.py \
  --input_file=$BERT_INPUT_FILE \
  --output_file=$BERT_OUTPUT_FILE \
  --vocab_file=$BERT_MODEL_DIR/vocab.txt \
  --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
  --init_checkpoint=$BERT_MODEL_DIR/bert_model.ckpt \
  --layers=$LAYERS \
  --max_seq_length=$MAX_SEQ_LEN \
  --batch_size=$BATCH_SIZE

export BERT_INPUT_FILE=$PROJ_ROOT/data/$DATASET_FOLDER_NAME/valid/src.txt
export BERT_OUTPUT_FILE=$PROJ_ROOT/data/$DATASET_FOLDER_NAME/valid/src.jsonl

python $BERT_DIR/extract_features.py \
  --input_file=$BERT_INPUT_FILE \
  --output_file=$BERT_OUTPUT_FILE \
  --vocab_file=$BERT_MODEL_DIR/vocab.txt \
  --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
  --init_checkpoint=$BERT_MODEL_DIR/bert_model.ckpt \
  --layers=$LAYERS \
  --max_seq_length=$MAX_SEQ_LEN \
  --batch_size=$BATCH_SIZE

export BERT_INPUT_FILE=$PROJ_ROOT/data/$DATASET_FOLDER_NAME/test/src.txt
export BERT_OUTPUT_FILE=$PROJ_ROOT/data/$DATASET_FOLDER_NAME/test/src.jsonl

python $BERT_DIR/extract_features.py \
  --input_file=$BERT_INPUT_FILE \
  --output_file=$BERT_OUTPUT_FILE \
  --vocab_file=$BERT_MODEL_DIR/vocab.txt \
  --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
  --init_checkpoint=$BERT_MODEL_DIR/bert_model.ckpt \
  --layers=$LAYERS \
  --max_seq_length=$MAX_SEQ_LEN \
  --batch_size=$BATCH_SIZE

# TARGET
export BERT_INPUT_FILE=$PROJ_ROOT/data/$DATASET_FOLDER_NAME/train/tgt.txt
export BERT_OUTPUT_FILE=$PROJ_ROOT/data/$DATASET_FOLDER_NAME/train/tgt.jsonl

python $BERT_DIR/extract_features.py \
  --input_file=$BERT_INPUT_FILE \
  --output_file=$BERT_OUTPUT_FILE \
  --vocab_file=$BERT_MODEL_DIR/vocab.txt \
  --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
  --init_checkpoint=$BERT_MODEL_DIR/bert_model.ckpt \
  --layers=$LAYERS \
  --max_seq_length=$MAX_SEQ_LEN \
  --batch_size=$BATCH_SIZE

export BERT_INPUT_FILE=$PROJ_ROOT/data/$DATASET_FOLDER_NAME/valid/tgt.txt
export BERT_OUTPUT_FILE=$PROJ_ROOT/data/$DATASET_FOLDER_NAME/valid/tgt.jsonl

python $BERT_DIR/extract_features.py \
  --input_file=$BERT_INPUT_FILE \
  --output_file=$BERT_OUTPUT_FILE \
  --vocab_file=$BERT_MODEL_DIR/vocab.txt \
  --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
  --init_checkpoint=$BERT_MODEL_DIR/bert_model.ckpt \
  --layers=$LAYERS \
  --max_seq_length=$MAX_SEQ_LEN \
  --batch_size=$BATCH_SIZE

export BERT_INPUT_FILE=$PROJ_ROOT/data/$DATASET_FOLDER_NAME/test/tgt.txt
export BERT_OUTPUT_FILE=$PROJ_ROOT/data/$DATASET_FOLDER_NAME/test/tgt.jsonl

python $BERT_DIR/extract_features.py \
  --input_file=$BERT_INPUT_FILE \
  --output_file=$BERT_OUTPUT_FILE \
  --vocab_file=$BERT_MODEL_DIR/vocab.txt \
  --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
  --init_checkpoint=$BERT_MODEL_DIR/bert_model.ckpt \
  --layers=$LAYERS \
  --max_seq_length=$MAX_SEQ_LEN \
  --batch_size=$BATCH_SIZE

