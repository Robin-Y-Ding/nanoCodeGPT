MAX_ITER=70000
WARMUP_ITER=700
EVAL_INTERVAL=200
SAVE_INTERVAL=10000
OUTPUT_DIR=out/thestack_fullpy_${MAX_ITER}
torchrun --standalone --nproc_per_node=4 train.py \
    --out_dir=$OUTPUT_DIR \
    --max_iters=$MAX_ITER \
    --warmup_iters=$WARMUP_ITER \
    --dataset=thestack/smol \
    --save_interval=$SAVE_INTERVAL \
    --eval_interval=$EVAL_INTERVAL \
    2>&1 | tee $OUTPUT_DIR/log_thestack_smol.txt