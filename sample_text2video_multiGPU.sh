
PROMPT="astronaut riding a horse" # OR: PROMPT="input/prompts.txt" for sampling multiple prompts
OUTDIR="results/"

BASE_PATH="models/base_t2v/model.ckpt"
CONFIG_PATH="models/base_t2v/model_config.ckpt"

NGPU=8

python -m torch.distributed.launch --nproc_per_node=$NGPU scripts/sample_text2video.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --prompt "$PROMPT" \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1000 \
    --ddp
