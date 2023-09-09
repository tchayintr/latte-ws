set -e

LANG=th
TEST_DATA=data/samples/best2010.test.sample10.seg.sl
MODEL_NAME=(
    best2010.latte.sample
)
MODEL_VERSION=(
    99
)
SAVE_DIR=(
    models/samples/decodes
)
DECODE_SAVE_DIR=(
    models/samples/decodes
)

CKPT_PATH=(
    # models/samples/best2010.sample/version_99/checkpoints/xxx.ckpt
)

DATA_LENGTH=${#CKPT_PATH[@]}
for ((i=0; i<$DATA_LENGTH; i++));
do
    python src/decode.py \
        --save-dir ${SAVE_DIR[$i]} \
        --test-file $TEST_DATA \
        --ckpt-path ${CKPT_PATH[$i]} \
        --model-name ${MODEL_NAME[$i]} \
        --model-version ${MODEL_VERSION[$i]} \
        --decode-save-dir ${DECODE_SAVE_DIR[$i]} \
        --normalize-unicode \
        --lang $LANG \
        --run bert \
        #
done
