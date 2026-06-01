# -*- coding: utf-8 -*-
import logging
import random
import os
import inspect

import torch
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from .moe_projection import TaskMoEProjection


@registry.register_model("mini_gpt4")
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model with self-expanding MoE projection.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/minigpt4_vicuna0.yaml",
        "pretrain_llama2": "configs/models/minigpt4_llama2.yaml",
    }

    @staticmethod
    def _filter_kwargs_for_cls(cls_or_fn, kwargs: dict) -> dict:
        try:
            sig = inspect.signature(cls_or_fn)
            valid = set(sig.parameters.keys())
        except Exception:
            return kwargs
        valid.discard("self")
        return {k: v for k, v in kwargs.items() if k in valid}

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        has_qformer=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym="\n",
        low_resource=False,
        device_8bit=0,
        lora_r=0,
        lora_target_modules=("q_proj", "v_proj"),
        lora_alpha=2,
        lora_dropout=0.05,
        linear=False,

        moe_init_experts: int = 1,

        moe_temp: float = 0.7,
        moe_router_noisy: bool = True,
        moe_router_noise_scale: float = 0.03,
        moe_router_use_gumbel: bool = True,
        moe_router_topk: int = 0,
        moe_train_mode: str = "soft",
        moe_eval_mode: str = "topk",
        moe_eval_topk: int = 2,

        moe_rd_dim: int = 128,
        moe_tau_proto: float = 0.10,
        moe_proto_momentum: float = 0.90,
        moe_update_proto: bool = True,
        moe_proto_init_random: bool = True,
        moe_min_proto_cnt: float = 10.0,

        moe_rd_var_target: float = 0.08,
        moe_rd_var_use_tokens: bool = True,

        moe_update_proto_in_detect: bool = False,
        moe_min_proto_cnt_detect: float = 1.0,

        moe_exp_threshold: float = 0.70,
        moe_min_outlier_frac: float = 0.20,
        moe_max_experts: int = 32,
        moe_expand_patience: int = 1,
        moe_expand_cooldown: int = 10,

        moe_expand_copy_from: str = "most_used",
        moe_expand_noise_std: float = 1e-2,

        moe_fuse_alpha: float = 0.7,
        moe_route_pool: str = "cls",
        moe_tie_break_eps: float = 1e-6,
        moe_router_init_std: float = 0.02,

        router_entropy_lambda: float = 5e-4,
        router_balance_lambda: float = 0.0,
        rd_lambda: float = 1e-4,
        align_lambda: float = 1e-3,
        kd_lambda: float = 5e-4,

        rd_var_lambda: float = 1e-3,

        moe_proto_update_mode: str = "hard",
        moe_proto_update_conf: float = 0.0,
        moe_proto_update_use_teacher: bool = True,

        moe_exp_use_zscore: bool = True,
        moe_exp_z_thresh: float = 2.5,
        moe_exp_z_ema_m: float = 0.99,
        moe_exp_min_cnt_for_z: float = 50.0,

        moe_expand_init_cnt: float = 2.0,
        moe_force_warmup_steps: int = 50,

        moe_proto_sep_enable: bool = True,
        moe_proto_sep_margin: float = 0.20,

        proto_sep_lambda: float = 1e-3,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.linear = linear

        self.router_entropy_lambda = float(router_entropy_lambda)
        self.router_balance_lambda = float(router_balance_lambda)
        self.rd_lambda = float(rd_lambda)
        self.align_lambda = float(align_lambda)
        self.kd_lambda = float(kd_lambda)
        self.rd_var_lambda = float(rd_var_lambda)
        self.proto_sep_lambda = float(proto_sep_lambda)

        print("Loading VIT")
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for _, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

            for _, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print("Loading VIT Done")

        self.has_qformer = has_qformer
        if self.has_qformer:
            print("Loading Q-Former")
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.load_from_pretrained(url_or_filename=q_former_model)

            if freeze_qformer:
                for _, param in self.Qformer.named_parameters():
                    param.requires_grad = False
                self.Qformer = self.Qformer.eval()
                self.Qformer.train = disabled_train
                self.query_tokens.requires_grad = False
                logging.info("freeze Qformer")

            img_f_dim = self.Qformer.config.hidden_size
            print("Loading Q-Former Done")
        else:
            img_f_dim = self.visual_encoder.num_features * 4
            print("Do not use Q-Former here.")

        print("Loading LLAMA")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(
            llama_model, local_files_only=True, use_fast=False, padding_side="right"
        )

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": device_8bit},
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_model.config.pad_token_id = self.llama_tokenizer.eos_token_id

        if lora_r > 0:
            if self.low_resource:
                self.llama_model = prepare_model_for_int8_training(self.llama_model)

            loraconfig = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=list(lora_target_modules),
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama_model = get_peft_model(self.llama_model, loraconfig)
            self.llama_model.print_trainable_parameters()
        else:
            for _, param in self.llama_model.named_parameters():
                param.requires_grad = False
        print("Loading LLAMA Done")

        moe_kwargs = dict(
            in_dim=img_f_dim,
            out_dim=self.llama_model.config.hidden_size,
            init_experts=int(moe_init_experts),

            temp=float(moe_temp),
            router_noisy=bool(moe_router_noisy),
            router_noise_scale=float(moe_router_noise_scale),
            router_use_gumbel=bool(moe_router_use_gumbel),
            router_topk=int(moe_router_topk),
            train_mode=str(moe_train_mode),
            eval_mode=str(moe_eval_mode),
            eval_topk=int(moe_eval_topk),

            rd_dim=int(moe_rd_dim),
            tau_proto=float(moe_tau_proto),
            proto_momentum=float(moe_proto_momentum),
            update_proto=bool(moe_update_proto),
            proto_init_random=bool(moe_proto_init_random),
            min_proto_cnt=float(moe_min_proto_cnt),

            rd_var_target=float(moe_rd_var_target),
            rd_var_use_tokens=bool(moe_rd_var_use_tokens),

            update_proto_in_detect=bool(moe_update_proto_in_detect),
            min_proto_cnt_detect=float(moe_min_proto_cnt_detect),

            exp_threshold=float(moe_exp_threshold),
            min_outlier_frac=float(moe_min_outlier_frac),
            max_experts=int(moe_max_experts),
            expand_patience=int(moe_expand_patience),
            expand_cooldown=int(moe_expand_cooldown),

            expand_copy_from=str(moe_expand_copy_from),
            expand_noise_std=float(moe_expand_noise_std),

            fuse_alpha=float(moe_fuse_alpha),
            route_pool=str(moe_route_pool),
            tie_break_eps=float(moe_tie_break_eps),
            router_init_std=float(moe_router_init_std),

            proto_update_mode=str(moe_proto_update_mode),
            proto_update_conf=float(moe_proto_update_conf),
            proto_update_use_teacher=bool(moe_proto_update_use_teacher),

            exp_use_zscore=bool(moe_exp_use_zscore),
            exp_z_thresh=float(moe_exp_z_thresh),
            exp_z_ema_m=float(moe_exp_z_ema_m),
            exp_min_cnt_for_z=float(moe_exp_min_cnt_for_z),

            expand_init_cnt=float(moe_expand_init_cnt),
            force_warmup_steps=int(moe_force_warmup_steps),

            proto_sep_enable=bool(moe_proto_sep_enable),
            proto_sep_margin=float(moe_proto_sep_margin),
        )
        moe_kwargs = self._filter_kwargs_for_cls(TaskMoEProjection.__init__, moe_kwargs)
        self.llama_proj = TaskMoEProjection(**moe_kwargs)

        try:
            self.llama_proj._default_force_warmup_steps = int(moe_force_warmup_steps)
        except Exception:
            pass

        if linear:
            self.linear_cls = torch.nn.Linear(4096, 200)
            self.order1 = MiniGPT4.get_classes_names()
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, "r") as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [p for p in raw_prompts if "<ImageHere>" in p]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print(f"Load {len(self.prompt_list)} training prompts")
            print(f"Prompt Example \n{random.choice(self.prompt_list)}")
        else:
            self.prompt_list = []

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image, task_id=None):
        device = image.device
        vision_dtype = next(self.visual_encoder.parameters()).dtype
        image = image.to(device=device, dtype=vision_dtype)

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))

            if self.has_qformer:
                image_atts = torch.ones(
                    image_embeds.size()[:-1],
                    dtype=torch.long,
                    device=device,
                )
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                inputs_llama = self.llama_proj(
                    query_output.last_hidden_state,
                    task_id=task_id,
                )
            else:
                image_embeds = image_embeds[:, 1:, :]
                bs, pn, hs = image_embeds.shape
                image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))
                inputs_llama = self.llama_proj(
                    image_embeds,
                    task_id=task_id,
                )

            atts_llama = torch.ones(
                inputs_llama.size()[:-1],
                dtype=torch.long,
                device=device,
            )

        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompts):
        if prompts:
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)

            for each_img_embed, each_prompt in zip(img_embeds, prompts):
                p_before, p_after = each_prompt.split("<ImageHere>")

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(img_embeds.device)

                p_before_embed = self.embed_tokens(p_before_tokens.input_ids)
                p_after_embed = self.embed_tokens(p_after_tokens.input_ids)
                wrapped_emb = torch.cat([p_before_embed, each_img_embed[None], p_after_embed], dim=1)
                emb_lists.append(wrapped_emb)

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)

            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, : emb_lens[i]] = emb
                wrapped_atts[i, : emb_lens[i]] = 1

            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        input_lens, cat_embs, cat_atts = [], [], []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(torch.cat([input_embs[i][:input_len], output_embs[i], input_embs[i][input_len:]]))
            cat_atts.append(torch.cat([input_atts[i][:input_len], output_atts[i], input_atts[i][input_len:]]))
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def eval_forward(self, image, task_id=None):
        img_embeds, _ = self.encode_img(image, task_id=task_id)
        img_feat = img_embeds.mean(dim=1).to(torch.float32)
        logits = self.linear_cls(img_feat)
        return logits

    def forward(self, samples):
        task_id = samples.get("task_id", None)

        if self.linear:
            image = samples["image"]
            vision_dtype = next(self.visual_encoder.parameters()).dtype
            image = image.to(device=image.device, dtype=vision_dtype)

            img_embeds, _ = self.encode_img(image, task_id=task_id)
            img_feat = img_embeds.mean(dim=1)
            logits = self.linear_cls(img_feat)

            targets = torch.zeros(logits.shape[0], device=image.device, dtype=torch.long)
            for i, t in enumerate(samples["answer"]):
                class_name = t.replace("This is a ", "").strip(".")
                targets[i] = self.order1.index(class_name)

            initial, increment = 20, 20
            max_target = int(targets.max().item())
            if max_target - initial < 0:
                end = initial
            else:
                end = initial + ((((max_target - initial) // increment) + 1) * increment)

            logits[:, end:].fill_(float("-inf"))
            loss = nn.functional.cross_entropy(logits, targets)
            return {"loss": loss}

        image = samples["image"]
        vision_dtype = next(self.visual_encoder.parameters()).dtype
        image = image.to(device=image.device, dtype=vision_dtype)

        img_embeds, atts_img = self.encode_img(image, task_id=task_id)

        if self.prompt_list:
            instruction = random.choice(self.prompt_list)
        else:
            instruction = samples["instruction_input"] if "instruction_input" in samples else None

        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, instruction)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["answer"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(image.device)

        batch_size = img_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds, attention_mask, input_lens = self.concat_emb_input_output(
            img_embeds, atts_img, to_regress_embeds, to_regress_tokens.attention_mask
        )
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

        part_targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = torch.ones(
            [inputs_embeds.shape[0], inputs_embeds.shape[1]],
            dtype=torch.long,
            device=image.device,
        ).fill_(-100)

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i] + 1: input_lens[i] + len(target) + 1] = target

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        main_loss = outputs.loss
        total_loss = main_loss

        aux = getattr(self.llama_proj, "last_aux", None)
        out = {"loss_main": main_loss}

        if isinstance(aux, dict):
            if "entropy" in aux and "loss_entropy" not in aux:
                aux["loss_entropy"] = aux["entropy"]
            if "loss_load_balance" in aux and "router_balance" not in aux:
                aux["router_balance"] = aux["loss_load_balance"]

            if "loss_entropy" in aux and self.router_entropy_lambda > 0:
                total_loss = total_loss + self.router_entropy_lambda * aux["loss_entropy"]
                out["loss_entropy"] = aux["loss_entropy"]

            if "router_balance" in aux and self.router_balance_lambda > 0:
                total_loss = total_loss + self.router_balance_lambda * aux["router_balance"]
                out["router_balance"] = aux["router_balance"]

            if "rd_loss" in aux and self.rd_lambda > 0:
                total_loss = total_loss + self.rd_lambda * aux["rd_loss"]
                out["rd_loss"] = aux["rd_loss"]

            if "rd_var_loss" in aux and self.rd_var_lambda > 0:
                total_loss = total_loss + self.rd_var_lambda * aux["rd_var_loss"]
                out["rd_var_loss"] = aux["rd_var_loss"]
                if "rd_std_min" in aux:
                    out["rd_std_min"] = aux["rd_std_min"]
                if "rd_std_mean" in aux:
                    out["rd_std_mean"] = aux["rd_std_mean"]

            if "proto_sep_loss" in aux and self.proto_sep_lambda > 0:
                total_loss = total_loss + self.proto_sep_lambda * aux["proto_sep_loss"]
                out["proto_sep_loss"] = aux["proto_sep_loss"]

            for k in ["proto_pair_sim_mean", "proto_pair_sim_max"]:
                if k in aux:
                    out[k] = aux[k]

            loss_align = aux.get("loss_align", None)
            loss_kd = aux.get("loss_kd", None)

            if (loss_align is not None) and (loss_kd is not None):
                same = False
                try:
                    same = (loss_align is loss_kd) or (loss_align.data_ptr() == loss_kd.data_ptr())
                except Exception:
                    same = (loss_align is loss_kd)

                if same:
                    lam = float(self.align_lambda) + float(self.kd_lambda)
                    if lam > 0:
                        total_loss = total_loss + lam * loss_align
                        out["loss_align"] = loss_align
                        out["loss_kd"] = loss_kd
                else:
                    if self.align_lambda > 0:
                        total_loss = total_loss + self.align_lambda * loss_align
                        out["loss_align"] = loss_align
                    if self.kd_lambda > 0:
                        total_loss = total_loss + self.kd_lambda * loss_kd
                        out["loss_kd"] = loss_kd
            else:
                if (loss_align is not None) and self.align_lambda > 0:
                    total_loss = total_loss + self.align_lambda * loss_align
                    out["loss_align"] = loss_align
                if (loss_kd is not None) and self.kd_lambda > 0:
                    total_loss = total_loss + self.kd_lambda * loss_kd
                    out["loss_kd"] = loss_kd

            for k in [
                "router_margin", "outlier_frac", "rd_max_sim_mean",
                "rd_max_sim_min", "rd_max_sim_max",
                "added_record", "outlier_streak", "cooldown_left",
                "proto_valid_num_detect", "proto_cnt_max", "proto_cnt_min", "num_experts"
            ]:
                if k in aux:
                    out[k] = aux[k]

            if "usage" in aux:
                out["router_usage"] = aux["usage"]

            self.llama_proj.last_aux = aux

        out["loss"] = total_loss
        return out

    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, "model"):  # lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get(
            "q_former_model",
            "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        )
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        has_qformer = cfg.get("has_qformer", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", "\n")

        lora_r = cfg.get("lora_r", 0)
        lora_alpha = cfg.get("lora_alpha", 2)
        lora_dropout = cfg.get("lora_dropout", 0.05)
        lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "v_proj"])
        linear = cfg.get("linear", False)

        moe_init_experts = cfg.get("moe_init_experts", 1)

        moe_temp = cfg.get("moe_temp", 0.7)
        moe_router_noisy = cfg.get("moe_router_noisy", True)
        moe_router_noise_scale = cfg.get("moe_router_noise_scale", 0.03)
        moe_router_use_gumbel = cfg.get("moe_router_use_gumbel", True)
        moe_router_topk = cfg.get("moe_router_topk", 0)
        moe_train_mode = cfg.get("moe_train_mode", "soft")
        moe_eval_mode = cfg.get("moe_eval_mode", "topk")
        moe_eval_topk = cfg.get("moe_eval_topk", 2)

        moe_rd_dim = cfg.get("moe_rd_dim", 128)
        moe_tau_proto = cfg.get("moe_tau_proto", 0.10)
        moe_proto_momentum = cfg.get("moe_proto_momentum", 0.90)
        moe_update_proto = cfg.get("moe_update_proto", True)
        moe_proto_init_random = cfg.get("moe_proto_init_random", True)
        moe_min_proto_cnt = cfg.get("moe_min_proto_cnt", 10.0)

        moe_rd_var_target = cfg.get("moe_rd_var_target", 0.08)
        moe_rd_var_use_tokens = cfg.get("moe_rd_var_use_tokens", True)

        moe_update_proto_in_detect = cfg.get("moe_update_proto_in_detect", False)
        moe_min_proto_cnt_detect = cfg.get("moe_min_proto_cnt_detect", 1.0)

        moe_exp_threshold = cfg.get("moe_exp_threshold", 0.70)
        moe_min_outlier_frac = cfg.get("moe_min_outlier_frac", 0.20)
        moe_max_experts = cfg.get("moe_max_experts", 32)
        moe_expand_patience = cfg.get("moe_expand_patience", 1)
        moe_expand_cooldown = cfg.get("moe_expand_cooldown", 10)

        moe_expand_copy_from = cfg.get("moe_expand_copy_from", "most_used")
        moe_expand_noise_std = cfg.get("moe_expand_noise_std", 1e-2)

        moe_fuse_alpha = cfg.get("moe_fuse_alpha", 0.7)
        moe_route_pool = cfg.get("moe_route_pool", "mean")
        moe_tie_break_eps = cfg.get("moe_tie_break_eps", 1e-6)
        moe_router_init_std = cfg.get("moe_router_init_std", 0.02)

        router_entropy_lambda = cfg.get("router_entropy_lambda", 5e-4)
        router_balance_lambda = cfg.get("router_balance_lambda", 0.0)
        rd_lambda = cfg.get("rd_lambda", 1e-4)
        align_lambda = cfg.get("align_lambda", 1e-3)
        kd_lambda = cfg.get("kd_lambda", 5e-4)
        rd_var_lambda = cfg.get("rd_var_lambda", 1e-3)

        moe_proto_update_mode = cfg.get("moe_proto_update_mode", "hard")
        moe_proto_update_conf = cfg.get("moe_proto_update_conf", 0.0)
        moe_proto_update_use_teacher = cfg.get("moe_proto_update_use_teacher", True)

        moe_exp_use_zscore = cfg.get("moe_exp_use_zscore", True)
        moe_exp_z_thresh = cfg.get("moe_exp_z_thresh", 2.5)
        moe_exp_z_ema_m = cfg.get("moe_exp_z_ema_m", 0.99)
        moe_exp_min_cnt_for_z = cfg.get("moe_exp_min_cnt_for_z", 50.0)

        moe_expand_init_cnt = cfg.get("moe_expand_init_cnt", 2.0)
        moe_force_warmup_steps = cfg.get("moe_force_warmup_steps", 300)

        moe_proto_sep_enable = cfg.get("moe_proto_sep_enable", True)
        moe_proto_sep_margin = cfg.get("moe_proto_sep_margin", 0.20)

        proto_sep_lambda = cfg.get("proto_sep_lambda", 1e-3)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            has_qformer=has_qformer,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            linear=linear,

            moe_init_experts=moe_init_experts,
            moe_temp=moe_temp,
            moe_router_noisy=moe_router_noisy,
            moe_router_noise_scale=moe_router_noise_scale,
            moe_router_use_gumbel=moe_router_use_gumbel,
            moe_router_topk=moe_router_topk,
            moe_train_mode=moe_train_mode,
            moe_eval_mode=moe_eval_mode,
            moe_eval_topk=moe_eval_topk,

            moe_rd_dim=moe_rd_dim,
            moe_tau_proto=moe_tau_proto,
            moe_proto_momentum=moe_proto_momentum,
            moe_update_proto=moe_update_proto,
            moe_proto_init_random=moe_proto_init_random,
            moe_min_proto_cnt=moe_min_proto_cnt,

            moe_rd_var_target=moe_rd_var_target,
            moe_rd_var_use_tokens=moe_rd_var_use_tokens,

            moe_update_proto_in_detect=moe_update_proto_in_detect,
            moe_min_proto_cnt_detect=moe_min_proto_cnt_detect,

            moe_exp_threshold=moe_exp_threshold,
            moe_min_outlier_frac=moe_min_outlier_frac,
            moe_max_experts=moe_max_experts,
            moe_expand_patience=moe_expand_patience,
            moe_expand_cooldown=moe_expand_cooldown,

            moe_expand_copy_from=moe_expand_copy_from,
            moe_expand_noise_std=moe_expand_noise_std,

            moe_fuse_alpha=moe_fuse_alpha,
            moe_route_pool=moe_route_pool,
            moe_tie_break_eps=moe_tie_break_eps,
            moe_router_init_std=moe_router_init_std,

            router_entropy_lambda=router_entropy_lambda,
            router_balance_lambda=router_balance_lambda,
            rd_lambda=rd_lambda,
            align_lambda=align_lambda,
            kd_lambda=kd_lambda,
            rd_var_lambda=rd_var_lambda,

            moe_proto_update_mode=moe_proto_update_mode,
            moe_proto_update_conf=moe_proto_update_conf,
            moe_proto_update_use_teacher=moe_proto_update_use_teacher,

            moe_exp_use_zscore=moe_exp_use_zscore,
            moe_exp_z_thresh=moe_exp_z_thresh,
            moe_exp_z_ema_m=moe_exp_z_ema_m,
            moe_exp_min_cnt_for_z=moe_exp_min_cnt_for_z,

            moe_expand_init_cnt=moe_expand_init_cnt,
            moe_force_warmup_steps=moe_force_warmup_steps,

            moe_proto_sep_enable=moe_proto_sep_enable,
            moe_proto_sep_margin=moe_proto_sep_margin,

            proto_sep_lambda=proto_sep_lambda,
        )

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print(f"Load BLIP2-LLM Checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            sd = ckpt["model"] if "model" in ckpt else ckpt

            if "llama_proj.weight" in sd or "llama_proj.bias" in sd:
                print("[MiniGPT4] Map single llama_proj -> llama_proj.experts.0 for compatibility")
                if "llama_proj.weight" in sd:
                    sd["llama_proj.experts.0.weight"] = sd.pop("llama_proj.weight")
                if "llama_proj.bias" in sd:
                    sd["llama_proj.experts.0.bias"] = sd.pop("llama_proj.bias")

            msg = model.load_state_dict(sd, strict=False)
            print("Missing/Unexpected keys:", msg)

        return model
