
mlx_lm.lora \
    --model Qwen/Qwen2.5-1.5B \
    --train \
    --data "./data" \
    --learning-rate 1e-5 \
    --iters 300 \
    --fine-tune-type full
