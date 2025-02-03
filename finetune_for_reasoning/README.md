
## the idea is to finetune a model to produce reasoning, in this case for arithmetic problems

#### github for code to generate test data (small arithmetic problems with reasoning):
    https://github.com/chrishayuk/chuk-math

#### how to use
Run the 0* files in order, first to create test data and then to train and test a model.

#### examples of running mlx commands

```bash
mlx_lm.generate --prompt "hello" --model meta-llama/Llama-3.2-1B-Instruct
```

```bash
mlx_lm.generate --prompt "hello" --model mlx-community/Llama-3.2-1B-Instruct-4bit
```

```bash
mlx_lm.chat --model mlx-community/Llama-3.2-1B-Instruct-4bit
```

```bash
mlx_lm.chat --model Qwen/Qwen2.5-1.5B
```

### sample query

what is (8895 + 631) / 22 + -952  # -519 is the answer
