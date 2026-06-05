# GPT-2 124M pretraining on OpenWebText (RTX 4060 Laptop 8GB)
# This is a slow but functional config for single 4060 8GB
# For serious pretraining, use 8xA100 with config/train_gpt2.py
# $ python train.py config/train_gpt2_4060.py

out_dir = 'out-gpt2-owt-4060'
eval_interval = 10
eval_iters = 200
log_interval = 1

wandb_log = True
wandb_run_name = 'gpt2-124M-4060'

dataset = 'openwebtext'

# GPT-2 124M architecture
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
dropout = 0.0
bias = False

# 4060 8GB VRAM config
# bs=4 * block_size=1024 * grad_accum=40 = 163,840 tokens/iter
# Original 8xA100: bs=12 * 1024 * 40 = 491,520 tokens/iter
batch_size = 4
gradient_accumulation_steps = 1*40

# Training schedule
# 600K iters * 163,840 tokens = ~98B tokens (vs original 300B)
# This is enough for a reasonable GPT-2 reproduction
max_iters = 600000
lr_decay_iters = 600000

# Learning rate (same as original GPT-2)
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 2000
weight_decay = 1e-1
beta2 = 0.99

# Windows does not support torch.compile
compile = False
