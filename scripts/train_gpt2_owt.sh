MAX_ITER=50000
WARMUP_ITER=500
OUTPUT_DIR=out/owt_${MAX_ITER}
torchrun --standalone --nproc_per_node=4 train.py \
    --out_dir=$OUTPUT_DIR \
    --max_iters=$MAX_ITER \
    --warmup_iters=$WARMUP_ITER \
    2>&1 | tee $OUTPUT_DIR/log_owt.txt