import os
import uuid


WANDB_RUN_ID_FILE = "wandb_run_id.txt"
WANDB_MODES = ("online", "offline")


def get_or_create_wandb_run_id(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    run_id_path = os.path.join(out_dir, WANDB_RUN_ID_FILE)
    if os.path.exists(run_id_path):
        with open(run_id_path, "r", encoding="utf-8") as f:
            run_id = f.read().strip()
        if run_id:
            return run_id

    run_id = uuid.uuid4().hex
    with open(run_id_path, "w", encoding="utf-8") as f:
        f.write(run_id + "\n")
    return run_id


def init_wandb(wandb, *, project, name, config, mode, out_dir):
    if mode not in WANDB_MODES:
        raise ValueError(f"wandb_mode must be one of {WANDB_MODES}, got {mode!r}")

    run_id = get_or_create_wandb_run_id(out_dir)
    return wandb.init(
        project=project,
        name=name,
        config=config,
        mode=mode,
        dir=out_dir,
        id=run_id,
        resume="allow",
    )
