set -e

DATA_ROOT_DIR=.
EXT_DIC_DATA=data/dict/hse-thai-lexitron.vocab.sl
NUM_GPUS=1
BATCH_SIZE=16
MAX_EPOCHS=20
PRETRAINED_MODEL=data/ptm/latte-mc-bert-bert-thai-ws
# PRETRAINED_MODEL=yacht/latte-mc-bert-base-thai-ws
SAVE_DIR=models/th
MODEL_VERSION=0
BERT_MODE=sum
LR=1e-3
BERT_LR=2e-5
GNN_LR=1e-3
LR_DECAY_RATE=0.99
LANG=th
GNN_TYPE=gat
CRITERION_TYPE=crf
ATTN_COMP_TYPE=wavg
MAX_TOKEN_LEN=12
METRIC_TYPE=word-bin-th
NODE_COMP_TYPE=none
DROPOUT=0.2
GRAPH_DROPOUT=0.2
ATTN_DROPOUT=0.2
ACC_GRAD_BATCH=8
GRADIENT_CLIP_VAL=5.0
SEED=112

# train/valid/test
# best2010
TRAIN_DATA=(
    data/th/best2010/best2010.train.shuf.seg.sl
)
VALID_DATA=(
    data/th/best2010/best2010.valid.shuf.seg.sl
)
TEST_DATA=(
    data/th/best2010/best2010.test.shuf.seg.sl
)
MODEL_NAME=(
    best2010.latte-mc-bert
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
        --num-gpus $NUM_GPUS \
        #
done
