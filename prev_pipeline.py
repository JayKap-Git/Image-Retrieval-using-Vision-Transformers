# """
# Milestone 2 — Image Understanding (Classification + Attributes + Retrieval)
# Single-file, end-to-end reference pipeline (updated & sanity-checked).

# Requires:
#     pip install torch torchvision timm scikit-learn pandas pyyaml pillow tqdm transformers

# Data layout (supports two CSV schemas):
#   data/
#     images/...                     # all image files
#     labels.csv                     # EITHER:
#                                    #  (A) wide:  path,class,color,material,condition,size
#                                    #  (B) packed:path,class,attributes,caption,instance_id
#                                    #      attributes="color:black;material:metal;condition:used;size:medium"
#     classes.txt                    # 10 lines, one class per line
#     attributes.yaml                # dict: color/material/condition/size vocab (includes "unknown")

# Models provided (choose via --model):
#   - tinyvit  -> timm: tiny_vit_11m_224
#   - swint    -> timm: swin_tiny_patch4_window7_224

# Improvements over earlier version:
#   • Robust CSV parsing (both schemas, case-insensitive columns & values)
#   • Attributes default to 'unknown' instead of being dropped
#   • Case-insensitive vocab lookup + humanized labels for retrieval phrases
#   • NaN-proof losses when a batch has only ignored labels
#   • Stratified split tolerant to rare classes (<2 samples)
#   • Built-in `checkdata` subcommand to sanity-check attributes & files
#   • Retrieval submodule with DistilBERT + contrastive projector
# """
# from __future__ import annotations
# import os
# import sys
# import json
# import math
# import time
# import yaml
# import random
# import argparse
# from dataclasses import dataclass
# from typing import Dict, List, Tuple, Optional

# import pandas as pd
# import numpy as np
# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# from torchvision import transforms

# from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

# import timm
# from tqdm import tqdm
# from collections import Counter

# # Text encoder for retrieval
# from transformers import DistilBertTokenizerFast, DistilBertModel

# # -----------------------------
# # Utilities & Reproducibility
# # -----------------------------

# def set_seed(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# def exists(p: str) -> bool:
#     return p is not None and len(p) > 0 and os.path.exists(p)


# def humanize_label(label: str) -> str:
#     """Make class labels readable and case-insensitive for display/phrases."""
#     if label is None:
#         return ""
#     s = str(label).replace("_", " ").strip()
#     s = " ".join(s.split())  # collapse whitespace
#     s = s.lower()
#     prefixes = [
#         "clothing",
#         "electronics accessories",
#         "personal care",
#         "cosmetics",
#         "food storage",
#         "stationary",
#         "tableware",
#         "travel",
#         "personal_care",
#         "electronics_accessories",
#         "food_storage",
#     ]
#     for p in prefixes:
#         if s.startswith(p + " "):
#             s = s[len(p) + 1:]
#             break
#     return s


# def parse_attr_string(attr: str) -> dict:
#     """Parse 'key:value;key:value' into a dict (lowercased, stripped)."""
#     out = {}
#     if attr is None:
#         return out
#     s = str(attr).strip()
#     if not s or s.lower() == "nan":
#         return out
#     parts = s.split(";")
#     for p in parts:
#         if ":" in p:
#             k, v = p.split(":", 1)
#             k = k.strip().lower()
#             v = v.strip().lower()
#             if k and v:
#                 out[k] = v
#     return out

# # -----------------------------
# # Vocabulary / Metadata loaders
# # -----------------------------
# @dataclass
# class Vocab:
#     classes: List[str]
#     colors: List[str]
#     materials: List[str]
#     conditions: List[str]
#     sizes: List[str]

#     @classmethod
#     def from_files(cls, classes_txt: str, attributes_yaml: str) -> "Vocab":
#         with open(classes_txt, "r", encoding="utf-8") as f:
#             classes = [ln.strip() for ln in f if ln.strip()]
#         with open(attributes_yaml, "r", encoding="utf-8") as f:
#             attrs = yaml.safe_load(f)
#         colors = list(attrs.get("color", []))
#         materials = list(attrs.get("material", []))
#         conditions = list(attrs.get("condition", []))
#         sizes = list(attrs.get("size", []))
#         return cls(classes, colors, materials, conditions, sizes)

#     def indexers(self) -> Dict[str, Dict[str, int]]:
#         # Case-insensitive lookup tables
#         return {
#             "class": {s.strip().lower(): i for i, s in enumerate(self.classes)},
#             "color": {s.strip().lower(): i for i, s in enumerate(self.colors)},
#             "material": {s.strip().lower(): i for i, s in enumerate(self.materials)},
#             "condition": {s.strip().lower(): i for i, s in enumerate(self.conditions)},
#             "size": {s.strip().lower(): i for i, s in enumerate(self.sizes)},
#         }

# # -----------------------------
# # Dataset
# # -----------------------------
# class MultiTaskImageDataset(Dataset):
#     def __init__(self, df: pd.DataFrame, root: str, vocab: Vocab, tfm: transforms.Compose):
#         self.df = df.reset_index(drop=True)
#         self.root = root
#         self.vocab = vocab
#         self.tfm = tfm
#         self.indexers = vocab.indexers()

#         def map_or_unknown(col: str, value: Optional[str]) -> int:
#             """Map token to id (case-insensitive). Blank/NaN/unseen -> 'unknown' if exists else -100."""
#             if value is None or pd.isna(value) or str(value).strip() == "" or str(value).strip().lower() == "nan":
#                 unk = self.indexers[col].get("unknown", None)
#                 return unk if unk is not None else -100
#             value = str(value).strip().lower()
#             idx = self.indexers[col].get(value, None)
#             if idx is None:
#                 unk = self.indexers[col].get("unknown", None)
#                 return unk if unk is not None else -100
#             return idx

#         # Detect schema
#         cols_lower = [c.lower() for c in self.df.columns]
#         has_wide = all(c in cols_lower for c in ["color","material","condition","size"])
#         has_attr_col = "attributes" in cols_lower

#         # Pre-encode targets
#         self.enc_class = []
#         self.enc_color = []
#         self.enc_material = []
#         self.enc_condition = []
#         self.enc_size = []

#         for _, row in self.df.iterrows():
#             cls_id = map_or_unknown("class", row.get("class"))
#             self.enc_class.append(cls_id)

#             color = material = condition = size = None
#             if has_wide:
#                 color = row.get("color")
#                 material = row.get("material")
#                 condition = row.get("condition")
#                 size = row.get("size")
#             elif has_attr_col:
#                 attrs = parse_attr_string(row.get("attributes"))
#                 color = attrs.get("color")
#                 material = attrs.get("material")
#                 condition = attrs.get("condition")
#                 size = attrs.get("size")

#             self.enc_color.append(map_or_unknown("color", color))
#             self.enc_material.append(map_or_unknown("material", material))
#             self.enc_condition.append(map_or_unknown("condition", condition))
#             self.enc_size.append(map_or_unknown("size", size))

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx: int):
#         row = self.df.iloc[idx]
#         img_path = row["path"]
#         full = img_path if os.path.isabs(img_path) else os.path.join(self.root, img_path)
#         with Image.open(full) as im:
#             im = im.convert("RGB")
#             x = self.tfm(im)
#         y = {
#             "class": torch.tensor(self.enc_class[idx], dtype=torch.long),
#             "color": torch.tensor(self.enc_color[idx], dtype=torch.long),
#             "material": torch.tensor(self.enc_material[idx], dtype=torch.long),
#             "condition": torch.tensor(self.enc_condition[idx], dtype=torch.long),
#             "size": torch.tensor(self.enc_size[idx], dtype=torch.long),
#         }
#         return x, y, full

# # -----------------------------
# # Transforms @ 224
# # -----------------------------
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]

# def make_transforms(res: int = 224):
#     train_tfm = transforms.Compose([
#         transforms.RandomResizedCrop(res, scale=(0.6, 1.0)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
#         transforms.ToTensor(),
#         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#     ])
#     test_tfm = transforms.Compose([
#         transforms.Resize(int(res * 1.14)),  # 256 for 224
#         transforms.CenterCrop(res),
#         transforms.ToTensor(),
#         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#     ])
#     return train_tfm, test_tfm

# # -----------------------------
# # Models: Backbones + Multi-heads
# # -----------------------------
# class MultiHeads(nn.Module):
#     def __init__(self, d: int, n_cls: int, n_color: int, n_mat: int, n_cond: int, n_size: int):
#         super().__init__()
#         self.cls = nn.Linear(d, n_cls)
#         self.color = nn.Linear(d, n_color)
#         self.mat = nn.Linear(d, n_mat)
#         self.cond = nn.Linear(d, n_cond)
#         self.size = nn.Linear(d, n_size)

#     def forward(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
#         return {
#             "class": self.cls(feats),
#             "color": self.color(feats),
#             "material": self.mat(feats),
#             "condition": self.cond(feats),
#             "size": self.size(feats),
#         }

# class MultiTaskNet(nn.Module):
#     def __init__(self, backbone_name: str, vocab: Vocab):
#         super().__init__()
#         self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
#         d = self.backbone.num_features
#         self.heads = MultiHeads(d,
#                                 n_cls=len(vocab.classes),
#                                 n_color=len(vocab.colors),
#                                 n_mat=len(vocab.materials),
#                                 n_cond=len(vocab.conditions),
#                                 n_size=len(vocab.sizes))

#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         feats = self.backbone(x)  # [B, d]
#         return self.heads(feats)

#     def extract_features(self, x: torch.Tensor) -> torch.Tensor:
#         return self.backbone(x)

# # -----------------------------
# # Losses & Metrics
# # -----------------------------
# @torch.no_grad()
# def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
#     mask = targets != -100
#     if mask.sum() == 0:
#         return float("nan")
#     pred = logits.argmax(dim=1)
#     acc = (pred[mask] == targets[mask]).float().mean().item()
#     return float(acc)


# def ce_loss(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
#     """Cross-entropy safe for batches with all-ignored targets."""
#     mask = targets != -100
#     if mask.sum() == 0:
#         return logits.sum() * 0.0
#     return F.cross_entropy(logits[mask], targets[mask], label_smoothing=label_smoothing)

# # -----------------------------
# # Training / Evaluation
# # -----------------------------
# @dataclass
# class TrainConfig:
#     model_name: str
#     data_root: str = "data"
#     labels_csv: str = "data/labels.csv"
#     classes_txt: str = "data/classes.txt"
#     attributes_yaml: str = "data/attributes.yaml"
#     out_dir: str = "outputs"

#     # Optimization
#     epochs: int = 80
#     batch_size: int = 64
#     lr: float = 2e-4
#     weight_decay: float = 5e-2
#     warmup_epochs: int = 8
#     label_smoothing_cls: float = 0.1

#     # Loss weights
#     w_color: float = 0.5
#     w_mat: float = 1.0
#     w_cond: float = 1.0
#     w_size: float = 1.0

#     # Misc
#     num_workers: int = 4
#     seed: int = 42
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"


# def cosine_scheduler(optimizer, base_lr, epochs, steps_per_epoch, warmup_epochs=0):
#     total_steps = epochs * steps_per_epoch
#     warmup_steps = warmup_epochs * steps_per_epoch

#     def lr_lambda(step):
#         if step < warmup_steps:
#             return (step + 1) / max(1, warmup_steps)
#         progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
#         return 0.5 * (1 + math.cos(math.pi * progress))

#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# def split_dataframe_stratified(df: pd.DataFrame, label_col: str = "class", train=0.7, val=0.15, test=0.15, seed: int = 42):
#     """Robust split handling rare classes (<2) by keeping them in TRAIN."""
#     from sklearn.model_selection import train_test_split

#     df = df.reset_index(drop=True)
#     counts = df[label_col].value_counts()

#     rare_labels = set(counts[counts < 2].index.tolist())
#     df_rare = df[df[label_col].isin(rare_labels)]
#     df_rem = df[~df[label_col].isin(rare_labels)]

#     if len(df_rem) == 0:
#         df_train = df.copy()
#         df_val = df.iloc[0:0].copy()
#         df_test = df.iloc[0:0].copy()
#         return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

#     try:
#         y = df_rem[label_col]
#         df_train_s, df_tmp, _, y_tmp = train_test_split(
#             df_rem, y, stratify=y, test_size=(1 - train), random_state=seed
#         )
#         relative_test = test / (val + test)
#         df_val_s, df_test_s, _, _ = train_test_split(
#             df_tmp, y_tmp, stratify=y_tmp, test_size=relative_test, random_state=seed
#         )
#     except Exception:
#         df_train_s, df_tmp = train_test_split(df_rem, test_size=(1 - train), random_state=seed)
#         relative_test = test / (val + test)
#         df_val_s, df_test_s = train_test_split(df_tmp, test_size=relative_test, random_state=seed)

#     df_train = pd.concat([df_train_s, df_rare], axis=0).sample(frac=1.0, random_state=seed)
#     df_val = df_val_s
#     df_test = df_test_s

#     return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


# def build_dataloaders(cfg: TrainConfig, vocab: Vocab, res: int = 224):
#     df = pd.read_csv(cfg.labels_csv)
#     # Normalize column names for case-insensitive parsing
#     df.columns = [str(c).strip().lower() for c in df.columns]

#     # Expand packed 'attributes' to wide columns if needed
#     needed = ["color","material","condition","size"]
#     has_wide = all(k in df.columns for k in needed)
#     if ("attributes" in df.columns) and (not has_wide):
#         parsed = df["attributes"].apply(parse_attr_string)
#         for k in needed:
#             df[k] = parsed.apply(lambda d: d.get(k, "unknown"))

#     assert set(["path","class"]).issubset(df.columns), "labels.csv must have at least columns: path,class"

#     tr, va, te = split_dataframe_stratified(df, label_col="class", seed=cfg.seed)
#     train_tfm, test_tfm = make_transforms(res)

#     ds_tr = MultiTaskImageDataset(tr, cfg.data_root, vocab, train_tfm)
#     ds_va = MultiTaskImageDataset(va, cfg.data_root, vocab, test_tfm)
#     ds_te = MultiTaskImageDataset(te, cfg.data_root, vocab, test_tfm)

#     dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
#     dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
#     dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
#     return dl_tr, dl_va, dl_te


# def train_one_epoch(model: MultiTaskNet, dl: DataLoader, opt, device, cfg: TrainConfig):
#     model.train()
#     loss_meter = 0.0
#     cls_acc_meter = 0.0
#     n_steps = 0

#     for x, y, _ in tqdm(dl, desc="train", leave=False):
#         x = x.to(device)
#         y_cls = y["class"].to(device)
#         y_color = y["color"].to(device)
#         y_mat = y["material"].to(device)
#         y_cond = y["condition"].to(device)
#         y_size = y["size"].to(device)

#         out = model(x)
#         loss = (
#             ce_loss(out["class"], y_cls, label_smoothing=cfg.label_smoothing_cls)
#             + cfg.w_color * ce_loss(out["color"], y_color)
#             + cfg.w_mat * ce_loss(out["material"], y_mat)
#             + cfg.w_cond * ce_loss(out["condition"], y_cond)
#             + cfg.w_size * ce_loss(out["size"], y_size)
#         )
#         if torch.isnan(loss):
#             print("[WARN] NaN loss encountered; replacing with 0 (likely an all-ignored head in this batch).")
#             loss = torch.zeros((), device=device)

#         opt.zero_grad()
#         loss.backward()
#         opt.step()

#         with torch.no_grad():
#             acc = masked_accuracy(out["class"], y_cls)
#         loss_meter += loss.item()
#         cls_acc_meter += 0.0 if math.isnan(acc) else acc
#         n_steps += 1

#     return loss_meter / max(1, n_steps), cls_acc_meter / max(1, n_steps)


# @torch.no_grad()
# def evaluate(model: MultiTaskNet, dl: DataLoader, device, vocab: Vocab, save_dir: Optional[str] = None, split_name: str = "val"):
#     model.eval()
#     all_logits = {k: [] for k in ["class","color","material","condition","size"]}
#     all_tgts = {k: [] for k in ["class","color","material","condition","size"]}

#     for x, y, _ in tqdm(dl, desc=f"eval-{split_name}", leave=False):
#         x = x.to(device)
#         out = model(x)
#         for k in all_logits:
#             all_logits[k].append(out[k].cpu())
#             all_tgts[k].append(y[k])

#     for k in all_logits:
#         all_logits[k] = torch.cat(all_logits[k], dim=0)
#         all_tgts[k] = torch.cat(all_tgts[k], dim=0)

#     # Count valid targets per head for diagnostics
#     valid_counts = {k: int((all_tgts[k] != -100).sum().item()) for k in all_tgts.keys()}

#     metrics = {}
#     # Classification metrics
#     cls_mask = all_tgts["class"] != -100
#     y_true = all_tgts["class"][cls_mask].numpy()
#     y_pred = all_logits["class"][cls_mask].argmax(dim=1).numpy()
#     metrics["class_acc"] = float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan")
#     metrics["class_f1_macro"] = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else float("nan")
#     metrics["class_f1_micro"] = float(f1_score(y_true, y_pred, average="micro")) if len(y_true) else float("nan")

#     # Confusion matrix
#     if len(y_true):
#         cm = confusion_matrix(y_true, y_pred, labels=list(range(len(vocab.classes))))
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#             np.save(os.path.join(save_dir, f"confmat_{split_name}.npy"), cm)

#     # Attribute accuracies
#     for head, name in [("color","color_acc"),("material","material_acc"),("condition","condition_acc"),("size","size_acc")]:
#         m = all_tgts[head] != -100
#         t = all_tgts[head][m].numpy()
#         p = all_logits[head][m].argmax(dim=1).numpy()
#         acc = float(accuracy_score(t, p)) if len(t) else float("nan")
#         f1m = float(f1_score(t, p, average="macro")) if len(t) else float("nan")
#         metrics[name] = acc
#         metrics[name.replace("acc","f1_macro")] = f1m

#     if save_dir:
#         # include valid counts to help debug NaNs
#         metrics_with_counts = {**metrics, **{f"n_valid_{k}": v for k, v in valid_counts.items()}}
#         with open(os.path.join(save_dir, f"metrics_{split_name}.json"), "w") as f:
#             json.dump(metrics_with_counts, f, indent=2)
#     return metrics


# def train_eval(cfg: TrainConfig):
#     set_seed(cfg.seed)

#     vocab = Vocab.from_files(cfg.classes_txt, cfg.attributes_yaml)
#     dl_tr, dl_va, dl_te = build_dataloaders(cfg, vocab)

#     device = torch.device(cfg.device)
#     model = MultiTaskNet(cfg.model_name, vocab).to(device)

#     opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
#     sched = cosine_scheduler(opt, cfg.lr, epochs=cfg.epochs, steps_per_epoch=len(dl_tr), warmup_epochs=cfg.warmup_epochs)

#     run_dir = os.path.join(cfg.out_dir, os.path.basename(cfg.model_name))
#     os.makedirs(run_dir, exist_ok=True)

#     best_val = -1.0
#     best_path = os.path.join(run_dir, "best.pt")
#     last_path = os.path.join(run_dir, "last.pt")

#     for epoch in range(1, cfg.epochs + 1):
#         t0 = time.time()
#         tr_loss, tr_acc = train_one_epoch(model, dl_tr, opt, device, cfg)
#         sched.step()

#         val_metrics = evaluate(model, dl_va, device, vocab, save_dir=run_dir, split_name="val")
#         val_score = val_metrics.get("class_f1_macro", float("nan"))

#         ckpt = {
#             "epoch": epoch,
#             "model": model.state_dict(),
#             "opt": opt.state_dict(),
#             "cfg": cfg.__dict__,
#             "val_metrics": val_metrics,
#         }
#         torch.save(ckpt, last_path)

#         if not math.isnan(val_score) and val_score > best_val:
#             best_val = val_score
#             torch.save(ckpt, best_path)

#         t1 = time.time()
#         print(f"Epoch {epoch:03d} | train_loss {tr_loss:.4f} | train_cls_acc {tr_acc:.4f} | val_f1_macro {val_score:.4f} | time {t1-t0:.1f}s")

#     print(f"Best val macro-F1: {best_val:.4f} | checkpoint: {best_path}")

#     # Final test using best checkpoint
#     if os.path.exists(best_path):
#         state = torch.load(best_path, map_location="cpu")
#         model.load_state_dict(state["model"])
#     test_metrics = evaluate(model, dl_te, device, vocab, save_dir=run_dir, split_name="test")
#     print("Test metrics:\n", json.dumps(test_metrics, indent=2))

# # -----------------------------
# # Data sanity-check subcommand
# # -----------------------------

# def run_checkdata(cfg: TrainConfig, check_images: bool = False, show: int = 10, top: int = 10):
#     vocab = Vocab.from_files(cfg.classes_txt, cfg.attributes_yaml)
#     allowed = {
#         "color": set(map(str.lower, vocab.colors)),
#         "material": set(map(str.lower, vocab.materials)),
#         "condition": set(map(str.lower, vocab.conditions)),
#         "size": set(map(str.lower, vocab.sizes)),
#     }
#     df = pd.read_csv(cfg.labels_csv)
#     df.columns = [str(c).strip().lower() for c in df.columns]

#     needed = ["color","material","condition","size"]
#     has_wide = all(k in df.columns for k in needed)
#     if ("attributes" in df.columns) and (not has_wide):
#         parsed = df["attributes"].apply(parse_attr_string)
#         for k in needed:
#             df[k] = parsed.apply(lambda d: d.get(k, "unknown"))
#         has_wide = True

#     if not has_wide:
#         print("ERROR: attributes not found (neither wide columns nor packed 'attributes').")
#         return

#     freq = {k: Counter() for k in needed}
#     unexpected = {k: Counter() for k in needed}
#     missing = {k: [] for k in needed}
#     unknown = {k: 0 for k in needed}

#     for i,row in df.iterrows():
#         for k in needed:
#             v = str(row.get(k, "")).strip().lower()
#             if v in ("", "nan"):
#                 missing[k].append(i)
#                 continue
#             if v == "unknown":
#                 unknown[k] += 1
#                 freq[k][v] += 1
#                 continue
#             if v not in allowed[k]:
#                 unexpected[k][v] += 1
#             freq[k][v] += 1

#     print(f"Total rows: {len(df)}")
#     print("\n=== Attribute coverage ===")
#     for k in needed:
#         print(f"- {k}: valid={sum(freq[k].values())} | missing={len(missing[k])} | unknown={unknown[k]}")

#     print("\n=== Unexpected tokens (not in attributes.yaml) ===")
#     for k in needed:
#         if unexpected[k]:
#             print(f"* {k}:")
#             for v,c in unexpected[k].most_common():
#                 print(f"    {v}: {c}")
#         else:
#             print(f"* {k}: none")

#     print("\n=== Top frequencies ===")
#     for k in needed:
#         print(f"* {k}:")
#         for v,c in freq[k].most_common(top):
#             print(f"    {v}: {c}")

#     if check_images:
#         miss = []
#         for i,row in df.iterrows():
#             p = row.get("path")
#             if p is None:
#                 miss.append((i, "<no path>")); continue
#             full = p if os.path.isabs(p) else os.path.join(cfg.data_root, p)
#             if not os.path.exists(full):
#                 miss.append((i, full))
#         if miss:
#             print(f"\nMissing image files: {len(miss)} (showing up to {show})")
#             for (idx, path) in miss[:show]:
#                 print(f"  row {idx}: {path}")
#         else:
#             print("\nAll image files found. ✔️")

# # -----------------------------
# # Retrieval submodule (optional)
# # -----------------------------
# class RetrievalHead(nn.Module):
#     def __init__(self, img_dim: int, txt_dim: int, proj_dim: int = 256):
#         super().__init__()
#         self.proj_img = nn.Sequential(
#             nn.Linear(img_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
#         )
#         self.proj_txt = nn.Sequential(
#             nn.Linear(txt_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
#         )
#         self.temperature = nn.Parameter(torch.tensor(0.07))

#     def forward(self, img_feats: torch.Tensor, txt_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         zi = F.normalize(self.proj_img(img_feats), dim=-1)
#         zt = F.normalize(self.proj_txt(txt_feats), dim=-1)
#         logits = zi @ zt.t() / self.temperature.clamp(min=1e-3)
#         return zi, zt, logits


# def info_nce_logits(logits: torch.Tensor, device) -> torch.Tensor:
#     targets = torch.arange(logits.size(0), device=device)
#     return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


# class TextEncoder(nn.Module):
#     def __init__(self, name: str = "distilbert-base-uncased"):
#         super().__init__()
#         self.tok = DistilBertTokenizerFast.from_pretrained(name)
#         self.enc = DistilBertModel.from_pretrained(name)
#         self.dim = self.enc.config.hidden_size

#     @torch.no_grad()
#     def encode(self, texts: List[str], device: str) -> torch.Tensor:
#         batch = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
#         out = self.enc(**batch)
#         cls = out.last_hidden_state[:, 0]
#         return cls

#     def forward(self, texts: List[str], device: str) -> torch.Tensor:
#         batch = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
#         out = self.enc(**batch)
#         return out.last_hidden_state[:, 0]


# @torch.no_grad()
# def build_phrases_from_row(row: pd.Series, vocab: Vocab) -> List[str]:
#     cls = humanize_label(row.get("class", ""))
#     color = str(row.get("color", "")).strip().lower()
#     material = str(row.get("material", "")).strip().lower()
#     condition = str(row.get("condition", "")).strip().lower()
#     size = str(row.get("size", "")).strip().lower()

#     phrases = []
#     if color and color != "nan" and color != "unknown":
#         phrases.append(f"{color} {cls}")
#     if material and material != "nan" and material != "unknown":
#         phrases.append(f"{material} {cls}")
#     if color and material and color != "unknown" and material != "unknown":
#         phrases.append(f"{color} {material} {cls}")
#     if condition and condition != "unknown" and size and size != "unknown":
#         phrases.append(f"{condition} {size} {cls}")
#     if not phrases:
#         phrases.append(cls)
#     return phrases


# def train_retrieval(cfg: TrainConfig, ckpt_path: str, max_pairs_per_image: int = 2, epochs: int = 10, lr: float = 1e-4, batch_size: int = 64):
#     set_seed(42)
#     device = torch.device(cfg.device)

#     vocab = Vocab.from_files(cfg.classes_txt, cfg.attributes_yaml)
#     df = pd.read_csv(cfg.labels_csv)
#     df.columns = [str(c).strip().lower() for c in df.columns]
#     needed = ["color","material","condition","size"]
#     has_wide = all(k in df.columns for k in needed)
#     if ("attributes" in df.columns) and (not has_wide):
#         parsed = df["attributes"].apply(parse_attr_string)
#         for k in needed:
#             df[k] = parsed.apply(lambda d: d.get(k, "unknown"))

#     model = MultiTaskNet(cfg.model_name, vocab).to(device)
#     state = torch.load(ckpt_path, map_location="cpu")
#     model.load_state_dict(state["model"])
#     model.eval()

#     txt = TextEncoder().to(device)
#     head = RetrievalHead(img_dim=model.backbone.num_features, txt_dim=txt.dim, proj_dim=256).to(device)

#     opt = torch.optim.AdamW(list(head.parameters()) + list(txt.parameters()), lr=lr, weight_decay=0.01)

#     pairs = []
#     for _, row in df.iterrows():
#         phrases = build_phrases_from_row(row, vocab)[:max_pairs_per_image]
#         img_path = row["path"]
#         full = img_path if os.path.isabs(img_path) else os.path.join(cfg.data_root, img_path)
#         for p in phrases:
#             pairs.append((full, p))

#     _, tfm = make_transforms(224)

#     def batch_iter(lst, bsz):
#         for i in range(0, len(lst), bsz):
#             yield lst[i:i+bsz]

#     for ep in range(1, epochs + 1):
#         random.shuffle(pairs)
#         losses = []
#         for chunk in tqdm(batch_iter(pairs, batch_size), total=math.ceil(len(pairs)/batch_size), desc=f"retrieval-train ep{ep}"):
#             ims = []
#             texts = []
#             for img_path, phrase in chunk:
#                 with Image.open(img_path) as im:
#                     im = im.convert("RGB")
#                     ims.append(tfm(im))
#                     texts.append(phrase)
#             ims = torch.stack(ims, dim=0).to(device)

#             with torch.no_grad():
#                 img_feats = model.extract_features(ims)

#             txt_feats = txt.forward(texts, device)
#             zi, zt, logits = head(img_feats, txt_feats)
#             loss = info_nce_logits(logits, device)

#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#             losses.append(loss.item())
#         print(f"Retrieval epoch {ep}: loss={np.mean(losses):.4f}")

#     save_dir = os.path.join(cfg.out_dir, os.path.basename(cfg.model_name))
#     os.makedirs(save_dir, exist_ok=True)
#     torch.save({"retrieval_head": head.state_dict(), "text_encoder": txt.state_dict()}, os.path.join(save_dir, "retrieval.pt"))
#     print("Saved retrieval artifacts.")


# @torch.no_grad()
# def demo_retrieve(cfg: TrainConfig, query: str, topk: int = 5):
#     device = torch.device(cfg.device)
#     vocab = Vocab.from_files(cfg.classes_txt, cfg.attributes_yaml)
#     df = pd.read_csv(cfg.labels_csv)
#     df.columns = [str(c).strip().lower() for c in df.columns]
#     needed = ["color","material","condition","size"]
#     has_wide = all(k in df.columns for k in needed)
#     if ("attributes" in df.columns) and (not has_wide):
#         parsed = df["attributes"].apply(parse_attr_string)
#         for k in needed:
#             df[k] = parsed.apply(lambda d: d.get(k, "unknown"))

#     run_dir = os.path.join(cfg.out_dir, os.path.basename(cfg.model_name))
#     best_path = os.path.join(run_dir, "best.pt")
#     assert os.path.exists(best_path), f"Missing best checkpoint at {best_path}"
#     state = torch.load(best_path, map_location="cpu")

#     model = MultiTaskNet(cfg.model_name, vocab).to(device)
#     model.load_state_dict(state["model"])
#     model.eval()

#     rpath = os.path.join(run_dir, "retrieval.pt")
#     assert os.path.exists(rpath), f"Missing retrieval head at {rpath} (train it first)"
#     rstate = torch.load(rpath, map_location="cpu")

#     txt = TextEncoder().to(device)
#     txt.load_state_dict(rstate["text_encoder"])  # type: ignore
#     head = RetrievalHead(model.backbone.num_features, txt.dim, 256).to(device)
#     head.load_state_dict(rstate["retrieval_head"])  # type: ignore

#     _, tfm = make_transforms(224)
#     img_paths = []
#     img_feats = []
#     for p in tqdm(df["path"].tolist(), desc="encode-gallery", leave=False):
#         full = p if os.path.isabs(p) else os.path.join(cfg.data_root, p)
#         with Image.open(full) as im:
#             im = im.convert("RGB")
#             x = tfm(im).unsqueeze(0).to(device)
#         f = model.extract_features(x)
#         img_paths.append(full)
#         img_feats.append(f)
#     img_feats = torch.cat(img_feats, dim=0)

#     qz = txt.encode([query], device)
#     zi, zt, _ = head(img_feats, qz)
#     scores = (zi @ zt.t()).squeeze(1).cpu().numpy()
#     top_idx = np.argsort(-scores)[:topk]

#     print("Query:", query)
#     print("Top-{}: ".format(topk))
#     for rank, idx in enumerate(top_idx, 1):
#         # Optionally display humanized class/attrs
#         row = df.iloc[idx]
#         cls = humanize_label(row.get("class", ""))
#         color = row.get("color", "?")
#         material = row.get("material", "?")
#         condition = row.get("condition", "?")
#         size = row.get("size", "?")
#         print(f"#{rank}: {img_paths[idx]} | score={scores[idx]:.4f} | {cls} | {color}, {material}, {condition}, {size}")

# # -----------------------------
# # CLI
# # -----------------------------
# MODELS = {
#     "tinyvit": "tiny_vit_11m_224",
#     "swint": "swin_tiny_patch4_window7_224",
# }


# def build_cfg(args) -> TrainConfig:
#     return TrainConfig(
#         model_name=MODELS[args.model],
#         data_root=args.data_root,
#         labels_csv=args.labels_csv,
#         classes_txt=args.classes_txt,
#         attributes_yaml=args.attributes_yaml,
#         out_dir=args.out_dir,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#         warmup_epochs=args.warmup,
#         label_smoothing_cls=args.label_smoothing,
#         w_color=args.w_color,
#         w_mat=args.w_mat,
#         w_cond=args.w_cond,
#         w_size=args.w_size,
#         num_workers=args.workers,
#         seed=args.seed,
#         device=("cuda" if torch.cuda.is_available() and not args.cpu else "cpu"),
#     )


# def main():
#     p = argparse.ArgumentParser(description="Milestone 2 Image Understanding Pipeline")
#     sub = p.add_subparsers(dest="cmd", required=True)

#     common = argparse.ArgumentParser(add_help=False)
#     common.add_argument("--model", choices=list(MODELS.keys()), default="tinyvit")
#     common.add_argument("--data_root", default="data")
#     common.add_argument("--labels_csv", default="data/labels.csv")
#     common.add_argument("--classes_txt", default="data/classes.txt")
#     common.add_argument("--attributes_yaml", default="data/attributes.yaml")
#     common.add_argument("--out_dir", default="outputs")
#     common.add_argument("--epochs", type=int, default=80)
#     common.add_argument("--batch_size", type=int, default=64)
#     common.add_argument("--lr", type=float, default=2e-4)
#     common.add_argument("--weight_decay", type=float, default=5e-2)
#     common.add_argument("--warmup", type=int, default=8)
#     common.add_argument("--label_smoothing", type=float, default=0.1)
#     common.add_argument("--w_color", type=float, default=0.5)
#     common.add_argument("--w_mat", type=float, default=1.0)
#     common.add_argument("--w_cond", type=float, default=1.0)
#     common.add_argument("--w_size", type=float, default=1.0)
#     common.add_argument("--workers", type=int, default=4)
#     common.add_argument("--seed", type=int, default=42)
#     common.add_argument("--cpu", action="store_true")

#     p_train = sub.add_parser("train", parents=[common])
#     p_eval = sub.add_parser("eval", parents=[common])

#     p_retr = sub.add_parser("retrieval", parents=[common])
#     p_retr.add_argument("--ckpt", required=False, help="Path to best/baseline checkpoint; defaults to outputs/<model>/best.pt")
#     p_retr.add_argument("--retr_epochs", type=int, default=10)
#     p_retr.add_argument("--retr_lr", type=float, default=1e-4)
#     p_retr.add_argument("--retr_batch_size", type=int, default=64)

#     p_demo = sub.add_parser("demo", parents=[common])
#     p_demo.add_argument("--query", required=True)
#     p_demo.add_argument("--topk", type=int, default=5)

#     # checkdata subcommand
#     p_check = sub.add_parser("checkdata", parents=[common])
#     p_check.add_argument("--check_images", action="store_true")
#     p_check.add_argument("--show", type=int, default=10)
#     p_check.add_argument("--top", type=int, default=10)

#     args = p.parse_args()
#     cfg = build_cfg(args)

#     if args.cmd == "train":
#         train_eval(cfg)
#     elif args.cmd == "eval":
#         vocab = Vocab.from_files(cfg.classes_txt, cfg.attributes_yaml)
#         dl_tr, dl_va, dl_te = build_dataloaders(cfg, vocab)
#         device = torch.device(cfg.device)
#         model = MultiTaskNet(cfg.model_name, vocab).to(device)
#         run_dir = os.path.join(cfg.out_dir, os.path.basename(cfg.model_name))
#         best_path = os.path.join(run_dir, "best.pt")
#         if os.path.exists(best_path):
#             state = torch.load(best_path, map_location="cpu")
#             model.load_state_dict(state["model"])
#         else:
#             print(f"WARN: best checkpoint not found at {best_path}; evaluating randomly initialized model.")
#         print("Val metrics:")
#         print(json.dumps(evaluate(model, dl_va, device, vocab, save_dir=run_dir, split_name="val"), indent=2))
#         print("Test metrics:")
#         print(json.dumps(evaluate(model, dl_te, device, vocab, save_dir=run_dir, split_name="test"), indent=2))
#     elif args.cmd == "retrieval":
#         run_dir = os.path.join(cfg.out_dir, os.path.basename(cfg.model_name))
#         ckpt = args.ckpt or os.path.join(run_dir, "best.pt")
#         train_retrieval(cfg, ckpt_path=ckpt, epochs=args.retr_epochs, lr=args.retr_lr, batch_size=args.retr_batch_size)
#     elif args.cmd == "demo":
#         demo_retrieve(cfg, query=args.query, topk=args.topk)
#     elif args.cmd == "checkdata":
#         run_checkdata(cfg, check_images=args.check_images, show=args.show, top=args.top)


# if __name__ == "__main__":
#     main()


