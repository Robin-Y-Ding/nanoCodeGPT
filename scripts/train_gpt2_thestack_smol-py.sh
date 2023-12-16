MAX_ITER=70000
WARMUP_ITER=700
OUTPUT_DIR=out/thestack_fullpy_${MAX_ITER}
torchrun --standalone --nproc_per_node=4 train.py \
    --out_dir=$OUTPUT_DIR \
    --max_iters=$MAX_ITER \
    --warmup_iters=$WARMUP_ITER \
    --dataset=thestack/smol-py \
    2>&1 | tee $OUTPUT_DIR/log_thestack_smol-py.txt