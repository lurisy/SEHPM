# -*- coding: utf-8 -*-
"""
runner_base.py  (Self-Expanding MoE Compatible)

要点：
1) 默认不强制“一任务一专家”，仅当 run_cfg.force_task_expert=True 才启用旧逻辑
2) optimizer 分组：base / llama_proj.experts / llama_proj.router_heads+norms / llama_proj.rd_proj
3) 若未来训练中途扩容导致参数数量变化，会给 warning（不自动重建，避免破坏 state）
4) checkpoint：保留全部 llama_proj.*（包括冻结专家/路由/原型buffer），其余仅保存可训练参数
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from minigpt4.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    is_dist_avail_and_initialized,
    main_process,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import is_url
from minigpt4.datasets.data_utils import (
    concat_datasets,
    reorg_datasets_by_split,
    ChainDataset,
)
from minigpt4.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)

from minigpt4.common import optims as _optims


@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default.
    """

    def __init__(self, cfg, task, model, datasets, job_id, task_id, **kwargs):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets

        # 注意：这里是“裸模型”，DDP wrap 由 self.model 属性延迟触发
        self._model = model
        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0
        self.task_id = task_id

        # ✅ 自扩展模式默认不强制“一任务一专家”
        if self.config.run_cfg.get("force_task_expert", False):
            self._ensure_task_expert()

        # 用于检测：optimizer 建立时的 moe 参数 fingerprint
        self._moe_param_fingerprint = None

        self.setup_output_dir()

    # ----------------- 基本属性 -----------------
    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)
        return self._device

    @property
    def use_distributed(self):
        return bool(self.config.run_cfg.distributed)

    def _model_device(self):
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def model(self):
        """
        延迟把 self._model 放到 device，并在需要时 DDP wrap。
        detect 阶段你应当使用 runner._model，避免这里触发 DDP。
        """
        if self._wrapped_model is None or self._model_device() != self.device:
            self._model = self._model.to(self.device)
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(self._model, device_ids=[self.config.run_cfg.gpu])
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    # ==========================
    # ✅ 关键：optimizer 分组 + MoE 路由/rd 单独更高 lr
    # ==========================
    @property
    def optimizer(self):
        if self._optimizer is None:
            base_lr = float(self.config.run_cfg.init_lr)
            wd = float(self.config.run_cfg.weight_decay)
            beta2 = self.config.run_cfg.get("beta2", 0.999)

            # 允许 yaml 指定绝对值；否则用倍率回退
            moe_expert_lr = float(self.config.run_cfg.get("moe_expert_lr", base_lr * 3.0))
            moe_router_lr = float(self.config.run_cfg.get("moe_router_lr", base_lr * 20.0))
            moe_rd_lr = float(self.config.run_cfg.get("moe_rd_lr", base_lr * 20.0))

            groups = {
                "base_wd": [], "base_nowd": [],
                "expert_wd": [], "expert_nowd": [],
                "router_wd": [], "router_nowd": [],
                "rd_wd": [], "rd_nowd": [],
            }

            num_parameters = 0

            # ✅ 注意：这里用 self.model.named_parameters()，DDP 时会带 module. 前缀
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue

                num_parameters += p.data.nelement()

                # no weight decay：bias / norm / 1D 参数
                is_nowd = (p.ndim < 2) or ("bias" in n) or ("ln" in n) or ("bn" in n)

                # --- MoE 投影层命名 ---
                # llama_proj.experts / llama_proj.router_heads / llama_proj.norms / llama_proj.rd_proj
                name = n[7:] if n.startswith("module.") else n

                if name.startswith("llama_proj.experts."):
                    key = "expert_nowd" if is_nowd else "expert_wd"

                elif name.startswith("llama_proj.router_heads.") or name.startswith("llama_proj.norms."):
                    key = "router_nowd" if is_nowd else "router_wd"

                elif name.startswith("llama_proj.rd_proj."):
                    key = "rd_nowd" if is_nowd else "rd_wd"

                else:
                    key = "base_nowd" if is_nowd else "base_wd"

                groups[key].append(p)

            logging.info("number of trainable parameters: %d" % num_parameters)

            optim_params = []

            # base
            if groups["base_wd"]:
                optim_params.append({"params": groups["base_wd"], "lr": base_lr, "weight_decay": wd})
            if groups["base_nowd"]:
                optim_params.append({"params": groups["base_nowd"], "lr": base_lr, "weight_decay": 0.0})

            # experts
            if groups["expert_wd"]:
                optim_params.append({"params": groups["expert_wd"], "lr": moe_expert_lr, "weight_decay": wd})
            if groups["expert_nowd"]:
                optim_params.append({"params": groups["expert_nowd"], "lr": moe_expert_lr, "weight_decay": 0.0})

            # router (router_heads + norms)
            if groups["router_wd"]:
                optim_params.append({"params": groups["router_wd"], "lr": moe_router_lr, "weight_decay": wd})
            if groups["router_nowd"]:
                optim_params.append({"params": groups["router_nowd"], "lr": moe_router_lr, "weight_decay": 0.0})

            # rd
            if groups["rd_wd"]:
                optim_params.append({"params": groups["rd_wd"], "lr": moe_rd_lr, "weight_decay": wd})
            if groups["rd_nowd"]:
                optim_params.append({"params": groups["rd_nowd"], "lr": moe_rd_lr, "weight_decay": 0.0})

            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=base_lr,  # 不影响 param_group 内部 lr
                weight_decay=wd,
                betas=(0.9, beta2),
            )

            self._moe_param_fingerprint = self._calc_moe_fingerprint()

        else:
            fp_now = self._calc_moe_fingerprint()
            if (self._moe_param_fingerprint is not None) and (fp_now != self._moe_param_fingerprint):
                logging.warning(
                    "[MoE] Detected llama_proj parameter change AFTER optimizer was created. "
                    "If you expand experts during training, new params will NOT be optimized unless you rebuild optimizer."
                )

        return self._optimizer

    def _calc_moe_fingerprint(self):
        """
        轻量指纹：统计 llama_proj 里 requires_grad 的参数数量/元素数。
        """
        try:
            m = self.unwrap_dist_model(self.model)
            if not hasattr(m, "llama_proj"):
                return None
            cnt = 0
            numel = 0
            for _, p in m.llama_proj.named_parameters():
                if p.requires_grad:
                    cnt += 1
                    numel += p.numel()
            return (cnt, numel)
        except Exception:
            return None

    @property
    def scaler(self):
        amp = self.config.run_cfg.get("amp", False)
        if amp and self._scaler is None:
            self._scaler = torch.cuda.amp.GradScaler()
        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            max_epoch = self.max_epoch
            min_lr = self.min_lr
            init_lr = self.init_lr

            decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)

            iters_per_epoch = self.config.run_cfg.get("iters_per_epoch", None)
            if iters_per_epoch is None:
                try:
                    if hasattr(self.dataloaders["train"], "loaders"):
                        iters_per_epoch = len(self.dataloaders["train"].loaders[0])
                    else:
                        iters_per_epoch = len(self.dataloaders["train"])
                except (AttributeError, TypeError):
                    iters_per_epoch = 10000

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                iters_per_epoch=iters_per_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    # ----------------- dataloader 构建 -----------------
    @property
    def dataloaders(self) -> dict:
        if self._dataloaders is None:
            logging.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) "
                "or chained (webdataset.DataPipeline)."
            )

            datasets = reorg_datasets_by_split(self.datasets)
            self.datasets = datasets

            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], (tuple, list)):
                    num_records = sum(
                        [
                            len(d)
                            if not isinstance(d, (wds.DataPipeline, ChainDataset))
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )
                else:
                    if hasattr(self.datasets[split_name], "__len__"):
                        num_records = len(self.datasets[split_name])
                    else:
                        num_records = -1
                        logging.info("Only a single wds.DataPipeline dataset, no __len__ attribute.")

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(num_records, split_name)
                    )

            split_names = sorted(self.datasets.keys())
            datasets = [self.datasets[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train if split == "train" else self.config.run_cfg.batch_size_eval
                for split in split_names
            ]

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, (tuple, list)):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        return int(self.config.run_cfg.get("log_freq", 50))

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])
        if len(valid_splits) == 0:
            logging.info("No validation splits found.")
        return valid_splits

    @property
    def test_splits(self):
        return self.config.run_cfg.get("test_splits", [])

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", [])
        if len(train_splits) == 0:
            logging.info("Empty train splits.")
        return train_splits

    @property
    def evaluate_only(self):
        return bool(self.config.run_cfg.evaluate)

    @property
    def use_dist_eval_sampler(self):
        return bool(self.config.run_cfg.get("use_dist_eval_sampler", True))

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def train_loader(self):
        return self.dataloaders["train"]

    # ----------------- 兼容旧逻辑：每 task 固定一个 expert -----------------
    def _ensure_task_expert(self):
        """
        每个 task 固定一个 expert：确保 num_experts >= task_id + 1。
        注意：自扩展模式建议关闭（run_cfg.force_task_expert=False）
        """
        tid = int(self.task_id)
        m = self._model  # 裸模型，避免 DDP wrap
        if m is None or not hasattr(m, "llama_proj"):
            return
        moe = m.llama_proj
        if not hasattr(moe, "experts"):
            return

        while len(moe.experts) <= tid:
            moe.add_expert(
                freeze_old_expert=True,
                freeze_old_router=False,
                freeze_old_norm=False,
            )

    # ----------------- 输出目录（按 task_id 区分） -----------------
    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id / str(self.task_id)
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir_" + str(self.task_id), str(result_dir))
        registry.register_path("output_dir_" + str(self.task_id), str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    # ----------------- 训练主循环 -----------------
    def train(self, ckpt_base=None, **kwargs):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()

        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if not self.evaluate_only:
                logging.info("Start training")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)

            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))
                    val_log = self.eval_epoch(split_name=split_name, cur_epoch=cur_epoch)
                    if val_log is not None and is_main_process():
                        assert "agg_metrics" in val_log, "No agg_metrics found in validation log."
                        agg_metrics = val_log["agg_metrics"]

                        if agg_metrics > best_agg_metric and split_name == "val":
                            best_epoch, best_agg_metric = cur_epoch, agg_metrics
                            self._save_checkpoint(cur_epoch, is_best=True, ckpt_base=ckpt_base)

                        val_log.update({"best_epoch": best_epoch})
                        self.log_stats(val_log, split_name)
            else:
                if not self.evaluate_only:
                    self._save_checkpoint(cur_epoch, is_best=False, ckpt_base=ckpt_base)

            if self.evaluate_only:
                break

            if self.config.run_cfg.distributed:
                dist.barrier()

        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()
        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name,
                    cur_epoch=cur_epoch,
                    skip_reload=skip_reload,
                )
            return test_logs

    def train_epoch(self, epoch):
        self.model.train()
        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(model=model, dataset=self.datasets[split_name])
        results = self.task.evaluation(model, data_loader)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )

    def unwrap_dist_model(self, model):
        return model.module if self.use_distributed else model

    # ----------------- dataloader 构造 -----------------
    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            if isinstance(dataset, (ChainDataset, wds.DataPipeline)):
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=False,
                )
                loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []
        for dataset, bsz, is_train, collate_fn in zip(datasets, batch_sizes, is_trains, collate_fns):
            if isinstance(dataset, (list, tuple)):
                if hasattr(dataset[0], "sample_ratio") and dataset_ratios is None:
                    dataset_ratios = [d.sample_ratio for d in dataset]
                loader = MultiIterLoader(
                    loaders=[_create_loader(d, num_workers, bsz, is_train, collate_fn[i]) for i, d in enumerate(dataset)],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)
            loaders.append(loader)

        return loaders

    # ----------------- Checkpoint 相关 -----------------
    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False, ckpt_base=None):
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()}
        state_dict = model_no_ddp.state_dict()

        # 保留所有 llama_proj.*（包括冻结的专家/路由/原型buffer）
        for k in list(state_dict.keys()):
            if k.startswith("llama_proj."):
                continue
            if k in param_grad_dic and not param_grad_dic[k]:
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }

        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        if self.max_epoch == cur_epoch + 1 or is_best:
            logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
            torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError:
            logging.warning("Key mismatch when loading checkpoint. Trying strict=False.")
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        if self.config.run_cfg.get("force_task_expert", False):
            self._ensure_task_expert()

        self.unwrap_dist_model(self.model).load_state_dict(state_dict, strict=False)

        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint and checkpoint["scaler"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint.get("epoch", -1) + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    # ----------------- 日志 -----------------
    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
