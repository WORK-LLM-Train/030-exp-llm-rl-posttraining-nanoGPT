# GPT-2 124M on Shakespeare (RTX 4060 Laptop 8GB)
# Quick test: verify 124M model trains correctly before full pretraining
# $ python train.py config/train_gpt2_shakespeare.py

out_dir = 'out-gpt2-shakespeare'
eval_interval = 200
eval_iters = 100
log_interval = 1
always_save_checkpoint = True

wandb_log = False
wandb_project = 'gpt2-shakespeare'
wandb_run_name = 'gpt2-124M-shakespeare'

# Shakespeare dataset (BPE tokenized with GPT-2 tokenizer)
dataset = 'shakespeare'

# GPT-2 124M architecture
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1
bias = True

# 4060 8GB VRAM optimized config
# bs=4 + block_size=512 -> ~3.5GB VRAM, fast per micro-step
# grad_accum=20 -> effective batch = 4 * 512 * 20 = 40,960 tokens/iter
batch_size = 4
block_size = 512
gradient_accumulation_steps = 20

# Fewer iters for test run
max_iters = 2000
lr_decay_iters = 2000

# Learning rate
learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 50
weight_decay = 1e-1

# Windows does not support torch.compile
compile = False
