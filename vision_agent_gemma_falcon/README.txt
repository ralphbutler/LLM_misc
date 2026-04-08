
git clone https://github.com/tiiuae/falcon-perception.git

Then install it with MLX support:

    cd falcon-perception
    pip install -e ".[mlx]"

We also had to upgrade einops to 0.8.2 to fix an MLX compatibility error.

We ran the gemma models under LM Studio on port 1234

The various run*.sh scripts show a few executions.
