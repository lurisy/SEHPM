import argparse
import os
import random
import re
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import init_distributed_mode, get_rank
from minigpt4.common.logger import setup_logger
from minigpt4.common.utils import now
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample

# Register modules
from minigpt4.datasets.builders import *   # noqa
from minigpt4.models import *              # noqa
from minigpt4.processors import *          # noqa
from minigpt4.runners import *             # noqa
from minigpt4.tasks import *               # noqa


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", default="train_configs/minigpt4_stage2_finetune.yaml")
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()


def setup_seeds(config):
    seed = int(config.run_cfg.seed) + int(get_rank())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    return registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

def map_single_proj_to_moe(sd):
    if "llama_proj.weight" in sd:
        sd["llama_proj.experts.0.weight"] = sd.pop("llama_proj.weight")
    if "llama_proj.bias" in sd:
        sd["llama_proj.experts.0.bias"] = sd.pop("llama_proj.bias")
    return sd

def grow_moe_to_match(model, sd):
    if not hasattr(model, "llama_proj"):
        return
    moe = model.llama_proj
    if not hasattr(moe, "experts"):
        return

    max_e = -1
    for k in sd.keys():
        m = re.match(r"^llama_proj\.experts\.(\d+)\.", k)
        if m:
            max_e = max(max_e, int(m.group(1)))

    if max_e < 0:
        return

    need = max_e + 1
    cur = len(moe.experts)
    if need > cur:
        if get_rank() == 0:
            print(f"[MoE] Expand experts {cur} → {need} (match ckpt)")
        for _ in range(need - cur):
            moe.add_expert(
                init_from=None,
                init_norm_from=None,
                freeze_old_expert=False,
                freeze_old_router=False,
                freeze_old_norm=False,
            )

def filter_incremental_state_dict(sd: dict) -> dict:
    keep = {}
    for k, v in sd.items():
        if k.startswith("llama_proj."):
            keep[k] = v
            continue

        lk = k.lower()
        if ("lora_" in lk) or (".lora_" in lk) or ("lora_a" in lk) or ("lora_b" in lk):
            keep[k] = v
            continue
    return keep


def filter_llama_proj_only(sd: dict) -> dict:
    return {k: v for k, v in sd.items() if k.startswith("llama_proj.")}

def _next_batch(loader):
    if hasattr(loader, "__next__"):
        return next(loader)
    if not hasattr(loader, "_tmp_iter_for_detect"):
        loader._tmp_iter_for_detect = iter(loader)
    return next(loader._tmp_iter_for_detect)


def detect_and_expand_moe(cfg, runner, task_id: int, ckpt_base: str, detect_batches: int = 50):
    raw_model = runner._model
    task = runner.task
    device = runner.device
    cuda_enabled = runner.cuda_enabled
    rank = get_rank()
    is_dist = dist.is_available() and dist.is_initialized()

    if not hasattr(raw_model, "llama_proj"):
        return False
    moe = raw_model.llama_proj
    if not hasattr(moe, "set_detecting_outlier"):
        return False

    train_loader = runner.train_loader
    added = False
    tmp_sync = os.path.join(ckpt_base, f"_tmp_expand_sync_task{task_id}.pth")

    raw_model.to(device)
    raw_model.train()

    iters_per_epoch = int(cfg.run_cfg.get("iters_per_epoch", 10000))
    inner_epoch = 0
    verbose = bool(cfg.run_cfg.get("moe_detect_verbose", False))

    bak_boot = None
    if len(moe.experts) == 1:
        bak_boot = {
            "exp_threshold": moe.exp_threshold,
            "min_outlier_frac": moe.min_outlier_frac,
            "expand_patience": moe.expand_patience,
            "expand_cooldown": moe.expand_cooldown,
            "expand_copy_from": getattr(moe, "expand_copy_from", "none"),
            "expand_noise_std": getattr(moe, "expand_noise_std", 0.0),
        }
        moe.exp_threshold = 0.60
        moe.min_outlier_frac = 0.01
        moe.expand_patience = 1
        moe.expand_cooldown = 0
        moe.expand_copy_from = "most_used"
        moe.expand_noise_std = 1e-2

    if verbose and rank == 0:
        print("[Detect] runner._model type =", type(getattr(runner, "_model", None)))
        print("[Detect] runner._wrapped_model type =", type(getattr(runner, "_wrapped_model", None)))
        print("[Detect] start task =", task_id, "detect_batches =", detect_batches)

    if (not is_dist) or rank == 0:
        moe.set_detecting_outlier(True)
        bak_update_proto = getattr(moe, "update_proto", None)

        consumed = 0
        try:
            if bak_update_proto is not None:
                moe.update_proto = False

            with torch.no_grad():
                for i in range(int(detect_batches)):
                    try:
                        samples = _next_batch(train_loader)
                    except StopIteration:
                        break

                    consumed += 1
                    samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                    samples.update({
                        "epoch": inner_epoch,
                        "num_iters_per_epoch": iters_per_epoch,
                        "iters": i,
                    })

                    samples["task_id"] = None

                    _ = task.train_step(model=raw_model, samples=samples)

                    aux = getattr(moe, "last_aux", {})
                    if verbose and isinstance(aux, dict) and (i % 10 == 0):
                        of = aux.get("outlier_frac", None)
                        ms = aux.get("rd_max_sim_mean", None)
                        st = aux.get("outlier_streak", None)
                        cd = aux.get("cooldown_left", None)
                        print(
                            f"[Detect][t{task_id}][{i:04d}] "
                            f"outlier_frac={of} max_sim_mean={ms} streak={st} cooldown={cd} "
                            f"E={len(moe.experts)}"
                        )

                    if isinstance(aux, dict) and aux.get("added_record", 0) == 1:
                        added = True
                        print(f"[Detect] Task {task_id}: add_expert at iter {i}, E={len(moe.experts)}")
                        break

        finally:
            if bak_update_proto is not None:
                moe.update_proto = bak_update_proto
            moe.set_detecting_outlier(False)

            if bak_boot is not None:
                moe.exp_threshold = bak_boot["exp_threshold"]
                moe.min_outlier_frac = bak_boot["min_outlier_frac"]
                moe.expand_patience = bak_boot["expand_patience"]
                moe.expand_cooldown = bak_boot["expand_cooldown"]
                moe.expand_copy_from = bak_boot["expand_copy_from"]
                moe.expand_noise_std = bak_boot["expand_noise_std"]

            if verbose and rank == 0:
                print(f"[Detect] Task {task_id}: consumed_batches={consumed}/{detect_batches}, added={added}, E={len(moe.experts)}")

        if is_dist and added:
            proj_sd = filter_llama_proj_only(raw_model.state_dict())
            torch.save({"model": proj_sd, "task_id": task_id}, tmp_sync)

    if is_dist:
        dist.barrier()
        if rank != 0:
            if os.path.exists(tmp_sync):
                ckpt = torch.load(tmp_sync, map_location="cpu")
                sd = ckpt["model"]
                grow_moe_to_match(raw_model, sd)
                raw_model.load_state_dict(sd, strict=False)

        dist.barrier()
        if rank == 0 and os.path.exists(tmp_sync):
            try:
                os.remove(tmp_sync)
            except Exception:
                pass

    raw_model.train()
    return added


def main():
    job_id = now()
    cfg = Config(parse_args())

    if "distributed" not in cfg.run_cfg:
        cfg.run_cfg["distributed"] = False

    init_distributed_mode(cfg.run_cfg)
    setup_logger()
    setup_seeds(cfg)
    cfg.pretty_print()

    ckpt_base = cfg.run_cfg.get("ckpt_base", "./models/checkpoint_dir")
    os.makedirs(ckpt_base, exist_ok=True)

    task = tasks.setup_task(cfg)
    cfg_o = cfg.get_o_config()
    task_num = cfg_o.task_num

    model = task.build_model(cfg)
    runner_cls = get_runner_class(cfg)

    base_ckpt = cfg.model_cfg.get("ckpt", None)
    if base_ckpt and os.path.exists(base_ckpt):
        if get_rank() == 0:
            print(f"[Init] Loading BASE ckpt: {base_ckpt}")
        ckpt = torch.load(base_ckpt, map_location="cpu")
        sd = ckpt["model"] if "model" in ckpt else ckpt
        sd = map_single_proj_to_moe(sd)
        grow_moe_to_match(model, sd)
        msg = model.load_state_dict(sd, strict=False)
        if get_rank() == 0:
            print("[Init] load_state:", msg)

    resume_ckpt = cfg.run_cfg.get("resume_ckpt_path", None)
    if resume_ckpt and os.path.exists(resume_ckpt):
        if get_rank() == 0:
            print(f"[Init] Loading RESUME ckpt: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        sd = ckpt["model"] if "model" in ckpt else ckpt
        sd = map_single_proj_to_moe(sd)
        grow_moe_to_match(model, sd)
        msg = model.load_state_dict(sd, strict=False)
        if get_rank() == 0:
            print("[Init] resume load_state:", msg)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    save_full_every = int(cfg.run_cfg.get("save_full_every", 0))

    for task_id in range(task_num):
        if get_rank() == 0:
            print("\n====================================")
            print(f"====== Training Task {task_id} ======")
            print("====================================")

        datasets = task.build_datasets(cfg, task_id=task_id)

        runner = runner_cls(
            cfg=cfg,
            job_id=job_id,
            task=task,
            model=model,
            datasets=datasets,
            task_id=task_id,
        )

        do_detect = bool(cfg.run_cfg.get("moe_detect_before_train", True))
        if do_detect and task_id >= int(cfg.run_cfg.get("moe_detect_start_task", 1)):
            detect_batches = int(cfg.run_cfg.get("moe_detect_batches", 200))
            added = detect_and_expand_moe(cfg, runner, task_id, ckpt_base, detect_batches)

            if added and hasattr(model, "llama_proj") and hasattr(model.llama_proj, "freeze_experts"):
                if bool(cfg.run_cfg.get("moe_freeze_old_experts", True)):
                    model.llama_proj.freeze_experts(True, only_old=True)
                    if get_rank() == 0:
                        print(f"[MoE] Freeze old experts, keep newest trainable. E={len(model.llama_proj.experts)}")

        runner.train(ckpt_base=ckpt_base)

        if get_rank() == 0:
            save_path = os.path.join(ckpt_base, f"checkpoint_{task_id}.pth")

            full_sd = model.state_dict()
            inc_sd = filter_incremental_state_dict(full_sd)

            torch.save({"model": inc_sd, "task_id": task_id}, save_path)
            print(f"[Task {task_id}] Incremental ckpt saved at: {save_path} (keys={len(inc_sd)})")

            if save_full_every > 0 and (task_id % save_full_every == 0):
                save_full = os.path.join(ckpt_base, f"checkpoint_full_{task_id}.pth")
                torch.save({"model": full_sd, "task_id": task_id}, save_full)
                print(f"[Task {task_id}] FULL ckpt saved at: {save_full}")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()


if __name__ == "__main__":
    main()
