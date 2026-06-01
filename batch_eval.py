# -*- coding: utf-8 -*-
import argparse
import os
import random
import re

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2

from minigpt4.datasets.builders import *  # noqa
from minigpt4.models import *             # noqa
from minigpt4.processors import *         # noqa
from minigpt4.runners import *            # noqa
from minigpt4.tasks import *              # noqa

from clip_base.datasets import build_cl_scenarios
import clip

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True,
                        help="path to configuration file.",
                        default="eval_configs/minigpt4_eval_all_tasks_imgr.yaml")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--task-id", type=int, default=0, help="which task you running")
    parser.add_argument("--ckpt-path", type=str, default='bad_path',
                        help="specify the path of ckpt for this task.")
    parser.add_argument("--txt-path", type=str, default='bad_path',
                        help="specify the path of results of this task.")
    parser.add_argument("--json-dir", type=str, required=True,
                        help="Path to JSON annotations directory")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    return parser.parse_args()

def setup_seeds(config):
    seed = int(config.run_cfg.seed) + int(get_rank())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

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
            idx = int(m.group(1))
            max_e = max(max_e, idx)

    if max_e < 0:
        return

    need = max_e + 1
    cur = len(moe.experts)

    if need > cur:
        print(f"[MoE] Expand experts {cur} → {need}")
        for _ in range(need - cur):
            moe.add_expert(
                init_from=None,
                init_norm_from=None,
                freeze_old_expert=False,
                freeze_old_router=False,
                freeze_old_norm=False,
            )

    if hasattr(moe, "router_heads") and hasattr(moe, "norms"):
        assert len(moe.experts) == len(moe.router_heads) == len(moe.norms), \
            f"MoE list mismatch: experts={len(moe.experts)}, heads={len(moe.router_heads)}, norms={len(moe.norms)}"

def _infer_llama_dtype(model):
    try:
        if hasattr(model, "llama_model"):
            return next(model.llama_model.parameters()).dtype
    except StopIteration:
        pass
    return torch.float16


def align_model_for_infer(model, device):
    dt = _infer_llama_dtype(model)
    model.to(device)

    if hasattr(model, "visual_encoder"):
        model.visual_encoder.to(device=device, dtype=dt)
    if hasattr(model, "ln_vision"):
        model.ln_vision.to(device=device, dtype=dt)

    if getattr(model, "has_qformer", False):
        if hasattr(model, "Qformer"):
            model.Qformer.to(device=device, dtype=dt)
        if hasattr(model, "query_tokens") and torch.is_tensor(model.query_tokens):
            model.query_tokens.data = model.query_tokens.data.to(device=device, dtype=dt)

    if hasattr(model, "llama_proj"):
        model.llama_proj.to(device=device, dtype=dt)

        moe = model.llama_proj
        if hasattr(moe, "rd_proto") and torch.is_tensor(moe.rd_proto):
            moe.rd_proto.data = moe.rd_proto.data.to(device=device, dtype=torch.float32)
        if hasattr(moe, "rd_proto_cnt") and torch.is_tensor(moe.rd_proto_cnt):
            moe.rd_proto_cnt.data = moe.rd_proto_cnt.data.to(device=device, dtype=torch.float32)

    return dt


def chat_answer_one(chat, CONV_VISION, img_1):
    chat_state = CONV_VISION.copy()
    img_list = []
    _ = chat.upload_img(img_1, chat_state, img_list)
    chat.ask('what is this photo of?', chat_state)

    out = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=1,
        max_new_tokens=12,
        max_length=256
    )

    if isinstance(out, (list, tuple)):
        if len(out) > 0 and isinstance(out[0], str):
            return out[0]
        if len(out) > 0 and isinstance(out[0], (list, tuple)) and len(out[0]) > 0:
            return out[0][0]
    return str(out)

def get_classes_names(
    path,
    class_order=None,
    expected_num_classes=None
):
    train_dir = os.path.join(path, "train")
    words_path = os.path.join(path, "README.txt")

    wnid_to_name = {}
    with open(words_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wnid, name = line.split(None, 1)
            wnid_to_name[wnid] = name.replace("_", " ")

    wnids = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )

    return [wnid_to_name[wnids[i]] for i in class_order]

conv_dict = {
    'pretrain_vicuna0': CONV_VISION_Vicuna0,
    'pretrain_llama2': CONV_VISION_LLama2
}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)
setup_seeds(cfg)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_config.ckpt = ""

moe_eval_mode = getattr(model_config, "moe_eval_mode", "soft")
moe_eval_topk = getattr(model_config, "moe_eval_topk", 0)

model_cls = registry.get_model_class(model_config.arch)
device = f'cuda:{args.gpu_id}'
model = model_cls.from_config(model_config).to(device)

if args.ckpt_path and os.path.exists(args.ckpt_path):
    print(f"[Eval] Loading ckpt from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt

    sd = map_single_proj_to_moe(sd)
    grow_moe_to_match(model, sd)

    msg = model.load_state_dict(sd, strict=False)
    print("[Eval] Missing / unexpected keys:", msg)

    align_model_for_infer(model, device)
else:
    print(f"[Eval] WARNING: ckpt {args.ckpt_path} not found, using random-initialized model!")
    align_model_for_infer(model, device)

model.eval()

if hasattr(model, "llama_proj"):
    moe = model.llama_proj
    moe.eval()

    if hasattr(moe, "set_eval_mode"):
        try:
            moe.set_eval_mode(moe_eval_mode, topk=moe_eval_topk)
        except Exception:
            pass

    if hasattr(moe, "set_force_warmup"):
        try:
            moe.set_force_warmup(None, steps=0)
        except Exception:
            pass

    if hasattr(moe, "set_force_idx"):
        try:
            moe.set_force_idx(None)
        except Exception:
            pass

    if hasattr(moe, "_force_left"):
        try:
            moe._force_left = 0
        except Exception:
            pass
    if hasattr(moe, "_force_idx"):
        try:
            moe._force_idx = None
        except Exception:
            pass

    if hasattr(moe, "set_detecting_outlier"):
        try:
            moe.set_detecting_outlier(False)
        except Exception:
            pass

    if hasattr(moe, "router_noisy"):
        moe.router_noisy = False

    if hasattr(moe, "router_topk"):
        moe.router_topk = 0

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}', task_id=None)
print('Initialization Finished')

cfg_o = cfg.get_o_config()
dataset_name = cfg_o.dataset
dataset_root = cfg_o.dataset_root
cifar_root= os.path.join(dataset_root, dataset_name)

class_order = cfg_o.class_order
class_num = cfg_o.class_num

device = f'cuda:{args.gpu_id}'
clip_model, clip_preprocess = clip.load("ViT-B/16", device=device, jit=False)
clip_model.eval()

eval_dataset, _ = build_cl_scenarios(
    cfg_o,
    is_train=False,
    cl_transforms=clip_preprocess
)

new_class_name = get_classes_names(cifar_root, class_order, expected_num_classes=class_num)


def build_class_bank(valid_names):
    prompts = [f"a photo of a {n}" for n in valid_names]
    tok = clip.tokenize(prompts).to(device)
    emb = clip_model.encode_text(tok)
    emb = F.normalize(emb, dim=-1)
    return emb


SIM_THRESHOLD = 0.18
_clip_query_cache = {}  # raw_msg(lower) -> (best_idx, best_sim)

with open(args.txt_path, 'w') as f, torch.inference_mode():
    try:
        eval_ds = eval_dataset[:args.task_id + 1]
    except Exception:
        eval_ds = eval_dataset

    eval_loader = DataLoader(eval_ds, batch_size=cfg_o.batch, shuffle=False)

    seen_classes = int(cfg_o.initial_increment) + int(args.task_id) * int(cfg_o.increment)
    seen_classes = max(1, min(class_num, seen_classes))

    names = new_class_name[:seen_classes]

    names_sorted = sorted(names, key=lambda x: len(x), reverse=True)
    name_patterns = [(n, re.compile(rf'\b{re.escape(n)}s?\b', flags=re.IGNORECASE)) for n in names_sorted]

    class_bank = build_class_bank(names)

    for inputs, targets, task_ids in tqdm(eval_loader):
        if torch.is_tensor(targets):
            mask = targets < seen_classes
            if mask.sum().item() == 0:
                continue
            inputs = inputs[mask]
            targets = targets[mask]
            task_ids = task_ids[mask] if torch.is_tensor(task_ids) else task_ids

        llm_message = []
        for i in range(inputs.shape[0]):
            img_1 = inputs[i:i+1].to(device)
            msg_i = chat_answer_one(chat, CONV_VISION, img_1)
            llm_message.append(msg_i)

        for i in range(inputs.shape[0]):
            if int(targets[i]) >= int(seen_classes):
                continue

            label = new_class_name[int(targets[i])]
            f.write(f'the label is {label}\n')

            raw_msg = str(llm_message[i]).strip()
            matched_name = None

            for cname, pattern in name_patterns:
                if pattern.search(raw_msg):
                    matched_name = cname
                    break

            if matched_name is None:
                key = raw_msg.strip().lower()
                if key in _clip_query_cache:
                    best_idx, best_sim = _clip_query_cache[key]
                else:
                    query_text = f"a photo of a {raw_msg}"
                    q_tok = clip.tokenize([query_text]).to(device)
                    q_emb = clip_model.encode_text(q_tok)
                    q_emb = F.normalize(q_emb, dim=-1)
                    sim = (q_emb @ class_bank.T).squeeze(0)
                    best_idx = int(torch.argmax(sim).item())
                    best_sim = float(sim[best_idx].item())
                    _clip_query_cache[key] = (best_idx, best_sim)

                if best_sim >= SIM_THRESHOLD:
                    matched_name = names[best_idx]

            if matched_name is not None:
                norm_msg = f"This is a photo of a {matched_name}."
            else:
                norm_msg = raw_msg

            f.write(f'msg: {norm_msg}\n')
