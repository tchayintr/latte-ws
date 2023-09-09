set -e

DATA_ROOT_DIR=.
TRAIN_DATA=data/samples/best2010.train.sample20.seg.sl
VALID_DATA=data/samples/best2010.valid.sample10.seg.sl
TEST_DATA=data/samples/best2010.test.sample10.seg.sl
EXT_DIC_DATA=data/dict/samples/hse-thai-lexitron.vocab.sample.sl
BATCH_SIZE=4
MAX_EPOCHS=2
PRETRAINED_MODEL=data/ptm/bert-base-multilingual-cased
SAVE_DIR=models/samples
MODEL_NAME=best2010.latte.sample
MODEL_VERSION=99
BERT_MODE=sum
LR=1e-3
BERT_LR=2e-5
GNN_LR=1e-3
LR_DECAY_RATE=0.9
LANG=th
CRITERION_TYPE=crf
METRIC_TYPE=word-bin-th
ATTN_COMP_TYPE=wavg
MAX_TOKEN_LEN=4
NODE_COMP_TYPE=none
MODEL_MAX_SEQ_LEN=64
DROPOUT=0.2
GRAPH_DROPOUT=0.1
ATTN_DROPOUT=0.2
ACC_GRAD_BATCH=1
GRADIENT_CLIP_VAL=5.0
N_GPUS=1
SEED=112

python3 src/train.py \
    --data-root-dir $DATA_ROOT_DIR \
    --train-file $TRAIN_DATA \
    --valid-file $VALID_DATA \
    --test-file $TEST_DATA \
    --batch-size $BATCH_SIZE \
    --max-epochs $MAX_EPOCHS \
    --pretrained-model $PRETRAINED_MODEL \
    --save-dir $SAVE_DIR \
    --model-name $MODEL_NAME \
    --model-version $MODEL_VERSION \
    --bert-mode $BERT_MODE \
    --lr $LR \
    --bert-lr $BERT_LR \
    --gnn-lr $GNN_LR \
    --lr-decay-rate $LR_DECAY_RATE \
    --optimized-decay \
    --scheduler \
    --lang $LANG \
    --normalize-unicode \
    --criterion-type $CRITERION_TYPE \
    --metric-type $METRIC_TYPE \
    --attn-comp-type $ATTN_COMP_TYPE \
    --max-token-length $MAX_TOKEN_LEN \
    --node-comp-type $NODE_COMP_TYPE \
    --ext-dic-file $EXT_DIC_DATA \
    --dropout $DROPOUT \
    --graph-dropout $GRAPH_DROPOUT \
    --attn-dropout $ATTN_DROPOUT \
    --normalize-unicode \
    --accumulate-grad-batches $ACC_GRAD_BATCH \
    --gradient_clip_val $GRADIENT_CLIP_VAL \
    --num-gpus $N_GPUS \
    --seed $SEED \
    #
