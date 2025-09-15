
## model_config.tsv contains MODEL names and info
# MODEL=gpt-4o-mini
# MODEL=claude-3-7-sonnet-latest
# MODEL=claude-sonnet-4-20250514
# MODEL=openai/phi-4-reasoning-plus-mlx
# MODEL=openai/rstar2-agent-14b-mlx
MODEL=openai/qwen3-next-80b-a3b-instruct

TIMEOUT=90

# YAML=capability_sampler.yaml
YAML=reasoning1.yaml


##### it costs about 13 cents to run this using sonnet-4 and reasoning1.yaml

/usr/bin/time -p \
    python3 benchmark_runner.py $YAML $MODEL --timeout $TIMEOUT | tee tempout1 2>&1

/usr/bin/time -p \
    python3 answer_extractor.py | tee tempout2 2>&1

