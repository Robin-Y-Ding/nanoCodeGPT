MAX_ITER=50000
WARMUP_ITER=500
EVAL_INTERVAL=200
SAVE_INTERVAL=10000
OUTPUT_DIR=out/owt_${MAX_ITER}
torchrun --standalone --nproc_per_node=4 train.py \
    --out_dir=$OUTPUT_DIR \
    --max_iters=$MAX_ITER \
    --warmup_iters=$WARMUP_ITER \
    --eval_interval=$EVAL_INTERVAL \
    --save_interval=$SAVE_INTERVAL \
    2>&1 | tee $OUTPUT_DIR/log_owt.txt