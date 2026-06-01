# -*- coding: utf-8 -*-
"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from minigpt4.common.dist_utils import (
    get_rank,
    get_world_size,
    is_main_process,
    is_dist_avail_and_initialized,
)
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()
        self.inst_id_key = "instance_id"
        self.cfg = None  # ✅ 保存 cfg（train_step 用）

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        # ✅ 关键：存 cfg，便于 train_step 读取 run_cfg 的 moe lambda
        self.cfg = cfg

        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg, task_id):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        """
        self.task_id = task_id
        datasets = dict()

        datasets_config = cfg.datasets_cfg
        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]
            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets(self.task_id)

            dataset["train"].task_id = task_id
            dataset["train"].name = name

            if "sample_ratio" in dataset_config:
                dataset["train"].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    # ----------------------------- utils -----------------------------
    @staticmethod
    def _as_float(v):
        if v is None:
            return None
        if torch.is_tensor(v):
            try:
                return float(v.detach().item())
            except Exception:
                return None
        try:
            return float(v)
        except Exception:
            return None

    @staticmethod
    def _alias_aux_keys(aux: dict) -> dict:
        """
        给不同版本的 MoE last_aux 做键名兼容（不会破坏新键名）：
        - entropy -> loss_entropy
        - loss_load_balance -> router_balance
        - loss_align -> loss_align（保留）并额外提供 loss_router 兼容别名
        """
        if not isinstance(aux, dict):
            return aux

        # entropy
        if "loss_entropy" not in aux and "entropy" in aux:
            aux["loss_entropy"] = aux["entropy"]

        # balance
        if "router_balance" not in aux and "loss_load_balance" in aux:
            aux["router_balance"] = aux["loss_load_balance"]

        # align/router loss
        if "loss_align" not in aux and "loss_router" in aux:
            aux["loss_align"] = aux["loss_router"]

        if "loss_router" not in aux and "loss_align" in aux:
            aux["loss_router"] = aux["loss_align"]

        return aux

    @staticmethod
    def _get_llama_proj(model):
        """
        兼容 DDP / 非 DDP：返回 llama_proj 或 None
        """
        if hasattr(model, "llama_proj"):
            return getattr(model, "llama_proj")
        if hasattr(model, "module") and hasattr(model.module, "llama_proj"):
            return getattr(model.module, "llama_proj")
        return None

    # ----------------------------- loss combiner (✅ 新增) -----------------------------
    def _get_moe_lambdas(self):
        """
        从 cfg.run_cfg 读取超参；没配就用默认值（偏保守，不容易炸）。
        """
        run_cfg = {}
        if self.cfg is not None and hasattr(self.cfg, "run_cfg"):
            run_cfg = self.cfg.run_cfg

        lam_align = float(run_cfg.get("moe_lambda_align", 1.0))
        lam_kd = float(run_cfg.get("moe_lambda_kd", 1.0))
        lam_balance = float(run_cfg.get("moe_lambda_balance", 0.05))
        lam_entropy = float(run_cfg.get("moe_lambda_entropy", 0.001))
        return lam_align, lam_kd, lam_balance, lam_entropy

    def _add_moe_aux_loss(self, loss, model):
        """
        ✅ 核心：把 MoE 的 last_aux 真正加进总 loss（否则 router 学不到分流信号）
        - loss_align: router_probs 对齐 teacher_probs（最关键）
        - loss_kd: topk/hard 时可选
        - router_balance: 防止塌缩（最小化 (1-balance)^2）
        - loss_entropy: 越大越不塌（最大化 entropy => 往 loss 里加 -entropy）
        """
        moe = self._get_llama_proj(model)
        if moe is None:
            return loss

        aux = getattr(moe, "last_aux", None)
        if not isinstance(aux, dict):
            return loss

        aux = self._alias_aux_keys(aux)
        lam_align, lam_kd, lam_balance, lam_entropy = self._get_moe_lambdas()

        # 1) align（最关键）
        if aux.get("loss_align", None) is not None:
            loss = loss + lam_align * aux["loss_align"]

        # 2) kd（可选）
        if aux.get("loss_kd", None) is not None:
            loss = loss + lam_kd * aux["loss_kd"]
        else:
            # 兼容一些旧键
            for kd_key in ["kd_loss", "loss_teacher", "teacher_loss"]:
                if aux.get(kd_key, None) is not None:
                    loss = loss + lam_kd * aux[kd_key]
                    break

        # 3) balance（理想≈1）
        if aux.get("router_balance", None) is not None:
            lb = aux["router_balance"]
            # lb 是 tensor；用 (1-lb)^2
            loss = loss + lam_balance * (1.0 - lb).pow(2)

        # 4) entropy（aux["loss_entropy"] 其实是 entropy；最大化 entropy => loss += -entropy）
        if aux.get("loss_entropy", None) is not None:
            ent = aux["loss_entropy"]
            loss = loss + lam_entropy * (-ent)

        return loss

    # ----------------------------- core steps -----------------------------
    def train_step(self, model, samples):
        # MiniGPT4.forward 返回 dict，loss 在 ["loss"]
        out = model(samples)
        loss = out["loss"]

        # ✅ 把 MoE 的 aux loss 加进总 loss（保证路由不塌 & 学会选路由）
        loss = self._add_moe_aux_loss(loss, model)

        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10
        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()
        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    # ----------------------------- train loop -----------------------------
    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # ====== MoE 监控项：全部“可选”，有就记，没有就保持 0 ======
        metric_logger.add_meter("num_experts", SmoothedValue(window_size=1, fmt="{value:.2f}"))

        metric_logger.add_meter("loss_entropy", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("router_margin", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("router_balance", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        metric_logger.add_meter("rd_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("outlier_frac", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("added_record", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("rd_max_sim_mean", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        metric_logger.add_meter("loss_align", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_router", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("router_acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_kd", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        init_update = {
            "lr": float(optimizer.param_groups[0]["lr"]),
            "loss": 0.0,
            "num_experts": 0.0,
            "loss_entropy": 0.0,
            "router_margin": 0.0,
            "router_balance": 0.0,
            "rd_loss": 0.0,
            "outlier_frac": 0.0,
            "added_record": 0.0,
            "rd_max_sim_mean": 0.0,
            "loss_align": 0.0,
            "loss_router": 0.0,
            "router_acc": 0.0,
            "loss_kd": 0.0,
        }
        metric_logger.update(**init_update)

        logging.info(f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch.")
        header = f"Train: data epoch: [{epoch}]"
        if start_iters is None:
            inner_epoch = epoch
        else:
            inner_epoch = start_iters // iters_per_epoch
            header = header + f"; inner epoch [{inner_epoch}]"

        # 兼容 MultiIterLoader / IterLoader / DataLoader
        if hasattr(data_loader, "loaders"):
            my_iter = len(data_loader.loaders[0])
        elif hasattr(data_loader, "__len__"):
            try:
                my_iter = len(data_loader)
            except TypeError:
                my_iter = int(iters_per_epoch)
        else:
            my_iter = int(iters_per_epoch)

        optimizer.zero_grad(set_to_none=True)

        for i in metric_logger.log_every(range(my_iter), log_freq, header):
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # ====== 读取 MoE last_aux（仅用于日志）======
            moe = self._get_llama_proj(model)
            aux = getattr(moe, "last_aux", None) if moe is not None else None

            if moe is not None and hasattr(moe, "experts"):
                metric_logger.update(num_experts=float(len(moe.experts)))

            if isinstance(aux, dict):
                aux = self._alias_aux_keys(aux)

                v = self._as_float(aux.get("loss_entropy", None))
                if v is not None:
                    metric_logger.update(loss_entropy=v)

                v = self._as_float(aux.get("router_margin", None))
                if v is not None:
                    metric_logger.update(router_margin=v)

                v = self._as_float(aux.get("router_balance", None))
                if v is not None:
                    metric_logger.update(router_balance=v)

                v = self._as_float(aux.get("rd_loss", None))
                if v is not None:
                    metric_logger.update(rd_loss=v)

                v = self._as_float(aux.get("outlier_frac", None))
                if v is not None:
                    metric_logger.update(outlier_frac=v)

                v = self._as_float(aux.get("added_record", None))
                if v is not None:
                    metric_logger.update(added_record=v)

                v = self._as_float(aux.get("rd_max_sim_mean", None))
                if v is not None:
                    metric_logger.update(rd_max_sim_mean=v)

                v = self._as_float(aux.get("loss_align", None))
                if v is not None:
                    metric_logger.update(loss_align=v)

                v = self._as_float(aux.get("loss_router", None))
                if v is not None:
                    metric_logger.update(loss_router=v)

                v = self._as_float(aux.get("router_acc", None))
                if v is not None:
                    metric_logger.update(router_acc=v)

                kd_val = None
                for kd_key in ["loss_kd", "kd_loss", "loss_teacher", "teacher_loss"]:
                    if kd_key in aux and aux[kd_key] is not None:
                        kd_val = self._as_float(aux[kd_key])
                        break
                if kd_val is not None:
                    metric_logger.update(loss_kd=kd_val)

            # backward
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # step
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            metric_logger.update(loss=float(loss.detach().item()))
            metric_logger.update(lr=float(optimizer.param_groups[0]["lr"]))

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            result = []
            for rank in range(get_world_size()):
                rf = os.path.join(result_dir, "%s_rank%d.json" % (filename, rank))
                res = json.load(open(rf, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
