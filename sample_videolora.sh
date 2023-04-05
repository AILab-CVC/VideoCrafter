
PROMPT="astronaut riding a horse" # OR: PROMPT="input/prompts.txt" for sampling multiple prompts
OUTDIR="results/videolora"

BASE_PATH="models/base_t2v/model.ckpt"
CONFIG_PATH="models/base_t2v/model_config.yaml"

# lora args
LORA_PATH="models/videolora/lora_001_Loving_Vincent_style.ckpt"
TAG=", Loving Vincent style"

python scripts/sample_text2video.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --prompt "$PROMPT" \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1000 \
    --show_denoising_progress \
    --inject_lora \
    --lora_path $LORA_PATH \
    --lora_trigger_word "$TAG" \
    --lora_scale 1.0
