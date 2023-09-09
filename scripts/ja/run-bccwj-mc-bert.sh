set -e

DATA_ROOT_DIR=.
EXT_DIC_DATA=data/dict/unidic_3_1-ipadic.vocab.sl
NUM_GPUS=1
BATCH_SIZE=32
MAX_EPOCHS=20
PRETRAINED_MODEL=data/ptm/latte-mc-bert-bert-japanese-ws
# PRETRAINED_MODEL=yacht/latte-mc-bert-base-japanese-ws
SAVE_DIR=models/ja
MODEL_VERSION=0
BERT_MODE=sum
LR=1e-3
BERT_LR=2e-5
GNN_LR=1e-3
LR_DECAY_RATE=0.90
LANG=ja
GNN_TYPE=gat
CRITERION_TYPE=crf
ATTN_COMP_TYPE=wavg
MAX_TOKEN_LEN=4
METRIC_TYPE=word-bin
NODE_COMP_TYPE=none
DROPOUT=0.2
GRAPH_DROPOUT=0.2
ATTN_DROPOUT=0.2
ACC_GRAD_BATCH=4
GRADIENT_CLIP_VAL=5.0
SEED=112

# train/valid/test
# bccwj
TRAIN_DATA=(
    data/ja/bccwj.train.seg.sl
)
VALID_DATA=(
    data/ja/bccwj.valid.seg.sl
)
TEST_DATA=(
    data/ja/bccwj.test.seg.sl
)
MODEL_NAME=(
    bccwj.latte-mc-bert
)
DATA_LENGTH=${#TRAIN_DATA[@]}
for ((i=0; i<$DATA_LENGTH; i++));
do
    python3 src/train.py \
        --data-root-dir $DATA_ROOT_DIR \
        --train-file ${TRAIN_DATA[$i]} \
        --valid-file ${VALID_DATA[$i]} \
        --test-file ${TEST_DATA[$i]} \
        --batch-size $BATCH_SIZE \
        --max-epochs $MAX_EPOCHS \
        --pretrained-model $PRETRAINED_MODEL \
        --save-dir $SAVE_DIR \
        --model-name ${MODEL_NAME[$i]} \
        --model-version $MODEL_VERSION \
        --bert-mode $BERT_MODE \
        --lr $LR \
        --bert-lr $BERT_LR \
        --gnn-lr $GNN_LR \
        --lr-decay-rate $LR_DECAY_RATE \
        --optimized-decay \
        --scheduler \
        --lang $LANG \
        --gnn-type $GNN_TYPE \
        --criterion-type $CRITERION_TYPE \
        --metric-type $METRIC_TYPE \
        --max-token-length $MAX_TOKEN_LEN \
        --attn-comp-type $ATTN_COMP_TYPE \
        --node-comp-type $NODE_COMP_TYPE \
        --dropout $DROPOUT \
        --graph-dropout $GRAPH_DROPOUT \
        --attn-dropout $ATTN_DROPOUT \
        --normalize-unicode \
        --ext-dic-file $EXT_DIC_DATA \
        --accumulate-grad-batches $ACC_GRAD_BATCH \
        --gradient-clip-val $GRADIENT_CLIP_VAL \
        --seed $SEED \
        --generate-unigram-node \
        --num-gpus $NUM_GPUS \
        #
done
