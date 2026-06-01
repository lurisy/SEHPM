# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskMoEProjection(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        init_experts: int = 1,

        temp: float = 0.7,
        router_noisy: bool = True,
        router_noise_scale: float = 0.01,
        router_use_gumbel: bool = True,
        router_topk: int = 0,
        train_mode: str = "soft",
        eval_mode: str = "soft",
        eval_topk: int = 0,

        rd_dim: int = 128,
        tau_proto: float = 0.10,
        proto_momentum: float = 0.90,
        update_proto: bool = True,
        proto_init_random: bool = True,
        min_proto_cnt: float = 10.0,
        min_proto_cnt_detect: Optional[float] = None,
        update_proto_in_detect: bool = False,

        rd_var_target: float = 0.08,
        rd_var_use_tokens: bool = True,

        exp_threshold: float = 0.70,
        min_outlier_frac: float = 0.20,
        max_experts: int = 32,
        expand_patience: int = 3,
        expand_cooldown: int = 20,

        expand_copy_from: str = "most_used",
        expand_noise_std: float = 1e-2,

        fuse_alpha: float = 0.7,
        route_pool: str = "cls",
        tie_break_eps: float = 1e-6,
        router_init_std: float = 0.02,
        usage_ema_momentum: float = 0.95,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.temp = float(temp)
        self.router_noisy = bool(router_noisy)
        self.router_noise_scale = float(router_noise_scale)
        self.router_use_gumbel = bool(router_use_gumbel)
        self.router_topk = int(router_topk)

        assert train_mode in ["soft", "hard", "task"]
        self.train_mode = train_mode
        assert eval_mode in ["soft", "hard", "topk"]
        self.eval_mode = eval_mode
        self.eval_topk = int(eval_topk)

        self.rd_dim = int(rd_dim)
        self.tau_proto = float(tau_proto)
        self.proto_momentum = float(proto_momentum)
        self.update_proto = bool(update_proto)
        self.proto_init_random = bool(proto_init_random)

        self.min_proto_cnt = float(min_proto_cnt)
        self.min_proto_cnt_detect = (
            float(min_proto_cnt_detect) if min_proto_cnt_detect is not None else float(min_proto_cnt)
        )
        self.update_proto_in_detect = bool(update_proto_in_detect)

        self.rd_var_target = float(rd_var_target)
        self.rd_var_use_tokens = bool(rd_var_use_tokens)

        self.exp_threshold = float(exp_threshold)
        self.min_outlier_frac = float(min_outlier_frac)
        self.max_experts = int(max_experts)
        self.expand_patience = int(expand_patience)
        self.expand_cooldown = int(expand_cooldown)

        self.expand_copy_from = str(expand_copy_from)
        assert self.expand_copy_from in ["none", "last", "most_used"]
        self.expand_noise_std = float(expand_noise_std)

        self.fuse_alpha = float(fuse_alpha)

        assert route_pool in ["mean", "cls"]
        self.route_pool = route_pool
        self.tie_break_eps = float(tie_break_eps)
        self.router_init_std = float(router_init_std)

        self.usage_ema_momentum = float(usage_ema_momentum)

        self.detecting_outlier: bool = False
        self._outlier_streak = 0
        self._cooldown_left = 0
        self._force_idx: Optional[int] = None

        self.rd_proj = nn.Sequential(
            nn.Linear(self.in_dim, self.rd_dim),
            nn.GELU(),
            nn.Linear(self.rd_dim, self.rd_dim),
        )

        self.register_buffer("rd_proto", torch.zeros(0, self.rd_dim))
        self.register_buffer("rd_proto_cnt", torch.zeros(0))

        self.register_buffer("ema_usage", torch.zeros(0))

        self.norms = nn.ModuleList()
        self.experts = nn.ModuleList()
        self.router_heads = nn.ModuleList()

        self.last_aux: Dict[str, Any] = {}

        for _ in range(int(init_experts)):
            self.add_expert(
                freeze_old_expert=False,
                freeze_old_router=False,
                freeze_old_norm=False,
                init_proto=None,
                init_from=None,
                init_norm_from=None,
                noise_std=0.0,
            )

    @torch.no_grad()
    def set_detecting_outlier(self, flag: bool):
        self.detecting_outlier = bool(flag)

    @torch.no_grad()
    def set_force_idx(self, idx: Optional[int]):
        self._force_idx = None if idx is None else int(idx)

    @torch.no_grad()
    def set_train_mode(self, mode: str):
        assert mode in ["soft", "hard", "task"]
        self.train_mode = mode

    @torch.no_grad()
    def set_eval_mode(self, mode: str, topk: Optional[int] = None):
        assert mode in ["soft", "hard", "topk"]
        self.eval_mode = mode
        if topk is not None:
            self.eval_topk = int(topk)

    @torch.no_grad()
    def set_temperature(self, t: float):
        self.temp = float(t)

    @torch.no_grad()
    def set_topk(self, k: int):
        self.router_topk = max(0, int(k))

    @staticmethod
    def _gumbel_like(t: torch.Tensor) -> torch.Tensor:
        u = torch.rand_like(t).clamp_(1e-6, 1 - 1e-6)
        return -torch.log(-torch.log(u))

    def _pool_route(self, x: torch.Tensor) -> torch.Tensor:
        if self.route_pool == "cls":
            return x[:, 0, :]
        return x.mean(dim=1)

    def _compute_rd(self, h_raw: torch.Tensor) -> torch.Tensor:
        rd = self.rd_proj(h_raw)
        return F.normalize(rd, dim=-1)

    def _ensure_proto_shape(self, E: int, device: torch.device):
        # proto
        if self.rd_proto.numel() == 0:
            self.rd_proto = torch.zeros(E, self.rd_dim, device=device, dtype=torch.float32)
            self.rd_proto_cnt = torch.zeros(E, device=device, dtype=torch.float32)
        elif self.rd_proto.shape[0] < E:
            pad = E - self.rd_proto.shape[0]
            self.rd_proto = torch.cat(
                [self.rd_proto, torch.zeros(pad, self.rd_dim, device=device, dtype=self.rd_proto.dtype)],
                dim=0
            )
            self.rd_proto_cnt = torch.cat(
                [self.rd_proto_cnt, torch.zeros(pad, device=device, dtype=self.rd_proto_cnt.dtype)],
                dim=0
            )

        if self.ema_usage.numel() == 0:
            self.ema_usage = torch.zeros(E, device=device, dtype=torch.float32)
        elif self.ema_usage.shape[0] < E:
            pad = E - self.ema_usage.shape[0]
            self.ema_usage = torch.cat(
                [self.ema_usage, torch.zeros(pad, device=device, dtype=self.ema_usage.dtype)],
                dim=0
            )

    @torch.no_grad()
    def _init_proto_random(self, idx: int, device: torch.device):
        if not self.proto_init_random:
            return
        if self.rd_proto_cnt[idx] <= 0:
            v = F.normalize(torch.randn(self.rd_dim, device=device, dtype=torch.float32), dim=0)
            self.rd_proto[idx].copy_(v)

    def _to_batch_task(self, task_id: Optional[torch.Tensor], B: int, device: torch.device) -> Optional[torch.Tensor]:
        if task_id is None:
            return None
        if torch.is_tensor(task_id):
            return task_id.view(-1).long().to(device)
        return torch.full((B,), int(task_id), device=device, dtype=torch.long)

    def _argmax_with_jitter(self, p: torch.Tensor) -> torch.Tensor:
        if self.tie_break_eps > 0:
            p = p + self.tie_break_eps * torch.rand_like(p)
        return torch.argmax(p, dim=-1)

    @staticmethod
    def _argmax_no_jitter(p: torch.Tensor) -> torch.Tensor:
        return torch.argmax(p, dim=-1)

    @staticmethod
    def _clone_linear(src: nn.Linear) -> nn.Linear:
        dst = nn.Linear(src.in_features, src.out_features, bias=(src.bias is not None))
        dst.load_state_dict(src.state_dict())
        return dst

    @staticmethod
    def _clone_layernorm(src: nn.LayerNorm) -> nn.LayerNorm:
        dst = nn.LayerNorm(
            src.normalized_shape, eps=src.eps, elementwise_affine=src.elementwise_affine
        )
        dst.load_state_dict(src.state_dict())
        return dst

    def add_expert(
        self,
        init_from: Optional[nn.Linear] = None,
        init_norm_from: Optional[nn.LayerNorm] = None,
        noise_std: float = 1e-3,
        freeze_old_expert: bool = True,
        freeze_old_router: bool = False,
        freeze_old_norm: bool = False,
        init_proto: Optional[torch.Tensor] = None,
    ) -> int:

        if len(self.experts) > 0:
            ref = next(self.experts[0].parameters())
        else:
            ref = next(self.rd_proj.parameters())
        dev, dtype = ref.device, ref.dtype

        if init_norm_from is None:
            norm_e = nn.LayerNorm(self.in_dim, eps=1e-6)
        else:
            norm_e = self._clone_layernorm(init_norm_from)
        norm_e = norm_e.to(device=dev, dtype=dtype)
        self.norms.append(norm_e)

        if init_from is None:
            expert_e = nn.Linear(self.in_dim, self.out_dim)
            nn.init.kaiming_uniform_(expert_e.weight, a=5 ** 0.5)
            if expert_e.bias is not None:
                nn.init.zeros_(expert_e.bias)
        else:
            expert_e = self._clone_linear(init_from)

        expert_e = expert_e.to(device=dev, dtype=dtype)
        if init_from is not None and noise_std > 0:
            with torch.no_grad():
                expert_e.weight.add_(noise_std * torch.randn_like(expert_e.weight))
                if expert_e.bias is not None:
                    expert_e.bias.add_(noise_std * torch.randn_like(expert_e.bias))
        self.experts.append(expert_e)

        head_e = nn.Linear(self.in_dim, 1)
        nn.init.normal_(head_e.weight, mean=0.0, std=self.router_init_std)
        nn.init.zeros_(head_e.bias)
        with torch.no_grad():
            head_e.bias.add_(1e-4 * (len(self.router_heads) - 0.5))
        head_e = head_e.to(device=dev, dtype=dtype)
        self.router_heads.append(head_e)

        if len(self.experts) > 1:
            if freeze_old_expert:
                for m in list(self.experts)[:-1]:
                    for p in m.parameters():
                        p.requires_grad = False
            if freeze_old_router:
                for m in list(self.router_heads)[:-1]:
                    for p in m.parameters():
                        p.requires_grad = False
            if freeze_old_norm:
                for m in list(self.norms)[:-1]:
                    for p in m.parameters():
                        p.requires_grad = False

        idx = len(self.experts) - 1
        self._ensure_proto_shape(len(self.experts), device=dev)

        if init_proto is not None:
            with torch.no_grad():
                proto = F.normalize(init_proto.view(-1).float(), dim=0)
                self.rd_proto[idx].copy_(proto.to(self.rd_proto.dtype))
                self.rd_proto_cnt[idx] = 0.0
        else:
            self._init_proto_random(idx, device=dev)
        return idx

    def _teacher_from_proto(self, rd: torch.Tensor) -> Optional[torch.Tensor]:
        E = len(self.experts)
        self._ensure_proto_shape(E, device=rd.device)

        valid = self.rd_proto_cnt[:E] >= self.min_proto_cnt
        if valid.sum().item() == 0:
            return None

        rd_fp = rd.float()
        proto_valid = self.rd_proto[:E][valid].float()
        sim = rd_fp @ proto_valid.t()
        teacher_valid = torch.softmax(sim / max(self.tau_proto, 1e-6), dim=-1)

        teacher = torch.zeros(rd.shape[0], E, device=rd.device, dtype=teacher_valid.dtype)
        teacher[:, valid] = teacher_valid
        teacher = teacher / (teacher.sum(dim=-1, keepdim=True) + 1e-9)
        return teacher

    @torch.no_grad()
    def _update_proto_ema_soft(self, rd: torch.Tensor, resp: torch.Tensor):
        E = len(self.experts)
        self._ensure_proto_shape(E, device=rd.device)

        rd_fp = rd.float()
        resp_fp = resp.float().clamp_min(0.0)

        w_sum = resp_fp.sum(dim=0) + 1e-9
        m = (resp_fp.t() @ rd_fp) / w_sum.unsqueeze(1)
        m = F.normalize(m, dim=-1)

        for e in range(E):
            if w_sum[e].item() > 1e-3:
                if self.rd_proto_cnt[e] <= 0:
                    self.rd_proto[e].copy_(m[e].to(self.rd_proto.dtype))
                else:
                    self.rd_proto[e].mul_(self.proto_momentum).add_(
                        (1 - self.proto_momentum) * m[e].to(self.rd_proto.dtype)
                    )
                    self.rd_proto[e].copy_(F.normalize(self.rd_proto[e], dim=0))
                self.rd_proto_cnt[e] += float(w_sum[e].item())

    def _pick_copy_source(self) -> Optional[int]:
        E = len(self.experts)
        if E <= 0:
            return None
        if self.expand_copy_from == "none":
            return None
        if self.expand_copy_from == "last":
            return E - 1
        if self.expand_copy_from == "most_used":
            if self.ema_usage.numel() >= E:
                return int(torch.argmax(self.ema_usage[:E]).item())
            return 0
        return None

    def _detect_and_expand(self, rd: torch.Tensor) -> torch.Tensor:
        B = rd.shape[0]
        E = len(self.experts)
        self._ensure_proto_shape(E, device=rd.device)

        self.last_aux["num_experts"] = int(E)
        self.last_aux["detecting_outlier"] = bool(self.detecting_outlier)
        cnt_all = self.rd_proto_cnt[:E].float()
        if cnt_all.numel() > 0:
            self.last_aux["proto_cnt_min"] = float(cnt_all.min().item())
            self.last_aux["proto_cnt_max"] = float(cnt_all.max().item())
        else:
            self.last_aux["proto_cnt_min"] = 0.0
            self.last_aux["proto_cnt_max"] = 0.0

        valid = self.rd_proto_cnt[:E] >= self.min_proto_cnt_detect
        self.last_aux["proto_valid_num_detect"] = int(valid.sum().item())

        if valid.sum().item() == 0:
            self.last_aux["added_record"] = 0
            self.last_aux["outlier_frac"] = 0.0
            self.last_aux["rd_max_sim_mean"] = 0.0
            self.last_aux["rd_max_sim_min"] = 0.0
            self.last_aux["rd_max_sim_max"] = 0.0
            self.last_aux["proto_valid_cnt"] = 0
            return torch.zeros(B, device=rd.device, dtype=torch.bool)

        rd_fp = rd.float()
        proto_valid = self.rd_proto[:E][valid].float()
        sim = rd_fp @ proto_valid.t()
        max_sim, _ = sim.max(dim=-1)

        outlier_mask = (max_sim < self.exp_threshold)
        frac = outlier_mask.float().mean().item()

        self.last_aux["proto_valid_cnt"] = int(valid.sum().item())
        self.last_aux["rd_max_sim_mean"] = float(max_sim.mean().item())
        self.last_aux["rd_max_sim_min"] = float(max_sim.min().item())
        self.last_aux["rd_max_sim_max"] = float(max_sim.max().item())
        self.last_aux["outlier_frac"] = float(frac)

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self.last_aux["cooldown_left"] = int(self._cooldown_left)
            self.last_aux["added_record"] = 0
            return outlier_mask

        if (frac >= self.min_outlier_frac) and (E < self.max_experts):
            self._outlier_streak += 1
        else:
            self._outlier_streak = 0
        self.last_aux["outlier_streak"] = int(self._outlier_streak)

        if self._outlier_streak >= self.expand_patience and E < self.max_experts:
            init_proto = rd[outlier_mask].mean(dim=0) if outlier_mask.any() else None

            src = self._pick_copy_source()
            init_from = self.experts[src] if (src is not None) else None
            init_norm_from = self.norms[src] if (src is not None) else None

            self.add_expert(
                init_from=init_from,
                init_norm_from=init_norm_from,
                noise_std=self.expand_noise_std if init_from is not None else 0.0,
                freeze_old_expert=True,
                freeze_old_router=False,
                freeze_old_norm=False,
                init_proto=init_proto,
            )

            self.last_aux["added_record"] = 1
            self.last_aux["new_expert_id"] = len(self.experts) - 1
            self.last_aux["copy_from"] = int(src) if src is not None else -1

            self._outlier_streak = 0
            self._cooldown_left = self.expand_cooldown
            self.last_aux["cooldown_left"] = int(self._cooldown_left)
        else:
            self.last_aux["added_record"] = 0

        return outlier_mask

    def forward(self, x: torch.Tensor, task_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device
        E = len(self.experts)

        if E > 0 and (len(self.norms) != E or len(self.router_heads) != E):
            ref_p = next(self.experts[0].parameters())
            ref_dev, ref_dtype = ref_p.device, ref_p.dtype

            while len(self.norms) < E:
                self.norms.append(self._clone_layernorm(self.norms[-1]).to(device=ref_dev, dtype=ref_dtype))
            while len(self.router_heads) < E:
                self.router_heads.append(self._clone_linear(self.router_heads[-1]).to(device=ref_dev, dtype=ref_dtype))

            if len(self.norms) > E:
                self.norms = nn.ModuleList(list(self.norms)[:E])
            if len(self.router_heads) > E:
                self.router_heads = nn.ModuleList(list(self.router_heads)[:E])

        self.last_aux = {"added_record": 0, "num_experts": int(E), "detecting_outlier": bool(self.detecting_outlier)}

        h_raw = self._pool_route(x)
        rd = self._compute_rd(h_raw)

        logits_list = [head(norm_e(h_raw)) for norm_e, head in zip(self.norms, self.router_heads)]
        logits = torch.cat(logits_list, dim=-1)

        if self.training and self.router_noisy and self.router_noise_scale > 0:
            noise = self._gumbel_like(logits) if self.router_use_gumbel else torch.randn_like(logits)
            logits = logits + self.router_noise_scale * noise

        router_probs = torch.softmax(logits / max(self.temp, 1e-6), dim=-1)

        with torch.no_grad():
            self._ensure_proto_shape(len(self.experts), device=device)
            cur = router_probs.detach().float().mean(dim=0)  # [E]
            m = self.usage_ema_momentum
            self.ema_usage[:len(self.experts)] = m * self.ema_usage[:len(self.experts)] + (1 - m) * cur

        teacher_probs = self._teacher_from_proto(rd)

        gate_probs = router_probs
        if self.training and self.router_topk > 0 and self.router_topk < E:
            k = max(1, min(self.router_topk, E))
            _, topi = torch.topk(gate_probs, k=k, dim=-1)
            mask = torch.zeros_like(gate_probs).scatter_(-1, topi, 1.0)
            gate_probs = gate_probs * mask
            gate_probs = gate_probs / (gate_probs.sum(dim=-1, keepdim=True) + 1e-9)

        if self.training and teacher_probs is not None:
            loss_align = F.kl_div(
                (router_probs + 1e-9).log(),
                teacher_probs.detach(),
                reduction="batchmean"
            )
            self.last_aux["loss_align"] = loss_align

            need_kd = False
            if (self.router_topk > 0 and self.router_topk < E):
                need_kd = True
            if self.train_mode in ["hard", "task"]:
                need_kd = True

            if need_kd:
                loss_kd = F.kl_div(
                    (gate_probs + 1e-9).log(),
                    teacher_probs.detach(),
                    reduction="batchmean"
                )
                self.last_aux["loss_kd"] = loss_kd

        task_b = self._to_batch_task(task_id, B, device=device)
        if task_b is not None:
            if (task_b.min() < 0) or (task_b.max() >= E):
                task_b = None

        if self._force_idx is not None:
            idx = int(self._force_idx)
            gate = F.one_hot(torch.full((B,), idx, device=device, dtype=torch.long), num_classes=E).float()
        elif self.training:
            if self.train_mode == "task" and task_b is not None:
                gate = F.one_hot(task_b, num_classes=E).float()
            elif self.train_mode == "hard":
                assign_idx = self._argmax_with_jitter(gate_probs)
                gate = F.one_hot(assign_idx, num_classes=E).float()
            else:
                gate = gate_probs
        else:
            if task_b is not None:
                gate = F.one_hot(task_b, num_classes=E).float()
            else:
                if teacher_probs is not None:
                    a = float(self.fuse_alpha)
                    fused = (router_probs.clamp_min(1e-9) ** a) * (teacher_probs.clamp_min(1e-9) ** (1 - a))
                    fused = fused / (fused.sum(dim=-1, keepdim=True) + 1e-9)
                else:
                    fused = router_probs

                if self.eval_mode == "soft":
                    gate = fused
                elif self.eval_mode == "hard":
                    assign_idx = self._argmax_with_jitter(fused)
                    gate = F.one_hot(assign_idx, num_classes=E).float()
                elif self.eval_mode == "topk":
                    k = max(1, min(self.eval_topk, E))
                    _, topi = torch.topk(fused, k=k, dim=-1)
                    mask = torch.zeros_like(fused).scatter_(-1, topi, 1.0)
                    gate = fused * mask
                    gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-9)
                else:
                    raise ValueError(f"Unknown eval_mode: {self.eval_mode}")

        if self.detecting_outlier:
            _ = self._detect_and_expand(rd)

            k2 = 2 if router_probs.size(-1) >= 2 else 1
            if k2 >= 2:
                top2, _ = router_probs.topk(k=k2, dim=-1)
                self.last_aux["router_margin"] = (top2[:, 0] - top2[:, 1]).mean()
            else:
                self.last_aux["router_margin"] = router_probs.new_tensor(0.0)

            if self.last_aux.get("added_record", 0) == 1:
                E = len(self.experts)
                self.last_aux["num_experts"] = int(E)

                logits_list = [head(norm_e(h_raw)) for norm_e, head in zip(self.norms, self.router_heads)]
                logits = torch.cat(logits_list, dim=-1)
                router_probs = torch.softmax(logits / max(self.temp, 1e-6), dim=-1)
                teacher_probs = self._teacher_from_proto(rd)

                if teacher_probs is not None:
                    a = float(self.fuse_alpha)
                    fused = (router_probs.clamp_min(1e-9) ** a) * (teacher_probs.clamp_min(1e-9) ** (1 - a))
                    fused = fused / (fused.sum(dim=-1, keepdim=True) + 1e-9)
                else:
                    fused = router_probs
                gate = fused

        if self.update_proto:
            allow_update = (
                (self.training and (not self.detecting_outlier)) or
                (self.detecting_outlier and self.update_proto_in_detect)
            )
            if allow_update:
                resp = router_probs.detach()
                self._update_proto_ema_soft(rd.detach(), resp)

        if gate.dtype != x.dtype:
            gate = gate.to(dtype=x.dtype)

        outs = torch.zeros((B, T, self.out_dim), device=device, dtype=x.dtype)
        for e_idx, expert_e in enumerate(self.experts):
            g = gate[:, e_idx].view(B, 1, 1)
            outs = outs + g * expert_e(x)

        if self.training:
            ent = -(router_probs * router_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
            self.last_aux["loss_entropy"] = ent

            with torch.no_grad():
                hard_assign = self._argmax_no_jitter(router_probs)
                hard = F.one_hot(hard_assign, num_classes=len(self.experts)).float()

            importance = router_probs.mean(dim=0)
            load = hard.mean(dim=0)
            lb = (importance * load).sum() * len(self.experts)

            self.last_aux["router_balance"] = lb
            self.last_aux["usage"] = load.detach()

            k2 = 2 if router_probs.size(-1) >= 2 else 1
            if k2 >= 2:
                top2, _ = router_probs.topk(k=k2, dim=-1)
                self.last_aux["router_margin"] = (top2[:, 0] - top2[:, 1]).mean()
            else:
                self.last_aux["router_margin"] = router_probs.new_tensor(0.0)

            if self.rd_var_target > 0:
                if self.rd_var_use_tokens:
                    rd_tok = self.rd_proj(x.reshape(-1, self.in_dim))
                    rd_tok = F.normalize(rd_tok, dim=-1)
                    std = rd_tok.float().std(dim=0, unbiased=False)
                else:
                    std = rd.float().std(dim=0, unbiased=False)

                rd_var_loss = F.relu(self.rd_var_target - std).mean()
                self.last_aux["rd_var_loss"] = rd_var_loss
                self.last_aux["rd_std_min"] = std.min()
                self.last_aux["rd_std_mean"] = std.mean()

            E_now = len(self.experts)
            if E_now >= 2:
                self._ensure_proto_shape(E_now, device=rd.device)
                proto = self.rd_proto[:E_now].detach()

                if teacher_probs is not None:
                    tgt_idx = torch.argmax(teacher_probs, dim=-1)
                else:
                    tgt_idx = hard_assign

                chosen = proto[tgt_idx.clamp(0, E_now - 1)]
                self.last_aux["rd_loss"] = (1.0 - (rd.float() * chosen.float()).sum(dim=-1)).mean()
            else:
                self.last_aux["rd_loss"] = rd.new_tensor(0.0)

        return outs
