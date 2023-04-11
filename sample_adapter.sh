PROMPT="An ostrich walking in the desert, photorealistic, 4k"
VIDEO="input/flamingo.mp4"
OUTDIR="results/"

NAME="video_adapter"
CONFIG_PATH="models/adapter_t2v_depth/model_config.yaml"
BASE_PATH="models/base_t2v/model.ckpt"
ADAPTER_PATH="models/adapter_t2v_depth/adapter.pth"

python scripts/sample_text2video_adapter.py \
    --seed 123 \
    --ckpt_path $BASE_PATH \
    --adapter_ckpt $ADAPTER_PATH \
    --base $CONFIG_PATH \
    --savedir $OUTDIR/$NAME \
    --bs 1 --height 256 --width 256 \
    --frame_stride -1 \
    --unconditional_guidance_scale 15.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --prompt "$PROMPT" \
    --video $VIDEO