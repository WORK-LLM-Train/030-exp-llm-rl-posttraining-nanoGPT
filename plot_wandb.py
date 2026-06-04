"""Plot metrics from a wandb offline run without needing Docker/login.
Usage: python plot_wandb.py [run_directory]
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import wandb

run_dir = sys.argv[1] if len(sys.argv) > 1 else "wandb/offline-run-20260526_173605-9dhqlccx"

api = wandb.Api()
run = api.run(str(Path(run_dir).absolute()))
history = run.history()

print(f"Metrics: {list(history.columns)}")
print(f"Steps: {len(history)}")

# --- loss ---
has_val = "val/loss" in history.columns
fig, axes = plt.subplots(1, 2 if has_val else 1, figsize=(12, 5))
if not has_val:
    axes = [axes]

axes[0].plot(history["_step"], history["train/loss"], label="train loss", alpha=0.8)
if has_val:
    axes[0].plot(history["_step"], history["val/loss"], label="val loss", alpha=0.8)
    axes[0].legend()
axes[0].set_xlabel("step")
axes[0].set_ylabel("loss")
axes[0].set_title("Loss")
axes[0].grid(True, alpha=0.3)

# --- lr ---
lr_cols = [c for c in history.columns if "lr" in c.lower()]
if lr_cols:
    for col in lr_cols:
        axes[1].plot(history["_step"], history[col], label=col)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("lr")
    axes[1].set_title("Learning Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
elif len(axes) > 1:
    axes[1].set_visible(False)

plt.tight_layout()
plt.savefig("wandb_plot.png", dpi=150)
print("Saved to wandb_plot.png")
plt.show()