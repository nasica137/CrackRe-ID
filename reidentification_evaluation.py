# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torchvision import models, transforms
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cityblock
import torch.nn as nn
import torch.nn.functional as F

# Geometry processing
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

################## HELPER FUNCTIONS ##################
def polygon_to_bbox(points):
    x = points[:, 0]
    y = points[:, 1]
    return [x.min(), y.min(), x.max(), y.max()]

def compare_bboxes(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter_area
    if union == 0:
        return 0
    return inter_area / union

def l2_normalize_np(v, eps=1e-12):
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def l2_normalize_torch(t, eps=1e-12):
    denom = t.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return t / denom

################## ATTENTION-BASED SIMILARITY ##################
class AttentionSimilarity(nn.Module):
    """
    Attention-like similarity between a query feature and memory features.
    Normalizes embeddings after projection and uses temperature scaling for logits.
    """
    def __init__(self, feature_dim, hidden_dim=2048, dropout_prob=0.2, temperature=0.1):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.k_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        # temperature for scaling cosine similarities (lower -> sharper softmax)
        self.temperature = float(temperature)
        # kept for backward compatibility but not used for logits
        self.scale = 1.0

    def encode_query(self, query_feat):
        """
        Returns L2-normalized query embedding after projection/norm/dropout.
        """
        device = next(self.parameters()).device
        if isinstance(query_feat, np.ndarray):
            query_feat = torch.tensor(query_feat, dtype=torch.float32)
        if query_feat.ndim == 1:
            query_feat = query_feat.unsqueeze(0)
        query_feat = query_feat.to(device)
        # Optional: L2-normalize raw input (safe if already normalized)
        query_feat = F.normalize(query_feat, p=2, dim=1)
        Q = self.query_proj(query_feat)
        Q = self.q_norm(Q)
        Q = self.dropout(Q)
        # L2-normalize projected embedding
        Q = F.normalize(Q, p=2, dim=1)
        return Q  # (B, hidden_dim)

    def encode_keys(self, memory_feats):
        """
        Returns L2-normalized key embeddings after projection/norm/dropout.
        """
        device = next(self.parameters()).device
        if isinstance(memory_feats, np.ndarray):
            memory_feats = torch.tensor(memory_feats, dtype=torch.float32)
        if memory_feats.ndim == 1:
            memory_feats = memory_feats.unsqueeze(0)
        memory_feats = memory_feats.to(device)
        # Optional: L2-normalize raw input
        memory_feats = F.normalize(memory_feats, p=2, dim=1)
        K = self.key_proj(memory_feats)
        K = self.k_norm(K)
        K = self.dropout(K)
        # L2-normalize projected embedding
        K = F.normalize(K, p=2, dim=1)
        return K  # (N, hidden_dim)

    def forward(self, query_feat, memory_feats):
        """
        Returns logits: cosine(Q, K) / temperature
        """
        Q = self.encode_query(query_feat)      # (B, H)
        K = self.encode_keys(memory_feats)     # (N, H)
        scores = (Q @ K.T) / self.temperature  # (B, N), cosine since Q,K are L2-normalized
        if scores.shape[0] == 1:
            return scores.squeeze(0)           # (N,)
        return scores                           # (B, N)

################## FLEXIBLE FEATURE EXTRACTOR ##################
class FlexibleFeatureExtractor:
    def __init__(self, backbone='dinov2'):
        self.backbone = backbone
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if backbone == 'dinov2':
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
            self.model.eval()
        else:
            raise ValueError("Unsupported backbone")

    def extract(self, img):
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # (1, D)
            # L2-normalize DINO feature
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            return features.squeeze(0).cpu().numpy().astype(np.float32)

################## GEOMETRY FEATURE EXTRACTOR (USING YOLO-SEG MASKS) ##################
class GeometryFeatureExtractor:
    def __init__(self, seg_weights_path, input_size=640, conf_threshold=0.1, mask_threshold=0.5):
        self.model = YOLO(seg_weights_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.mask_threshold = mask_threshold

    def pad_to_square_with_meta(self, img_bgr, new_size=640, color=(0, 0, 0)):
        h, w = img_bgr.shape[:2]
        scale = min(new_size / h, new_size / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_top = (new_size - nh) // 2
        pad_bottom = new_size - nh - pad_top
        pad_left = (new_size - nw) // 2
        pad_right = new_size - nw - pad_left
        img_padded = cv2.copyMakeBorder(
            img_resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=color
        )
        meta = {
            "pad_top": pad_top,
            "pad_left": pad_left,
            "valid_h": nh,
            "valid_w": nw,
            "input_size": new_size,
            "orig_h": h,
            "orig_w": w
        }
        return img_padded, meta

    def _fallback_mask(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_clean = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        return (mask_clean > 0).astype(np.uint8)

    def get_mask(self, img_pil):
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        padded_bgr, meta = self.pad_to_square_with_meta(img_bgr, new_size=self.input_size, color=(0, 0, 0))
        results = self.model.predict(
            source=padded_bgr, imgsz=self.input_size, conf=self.conf_threshold,
            save=False, verbose=False
        )
        res = results[0]

        if res.masks is None or res.masks.data is None or len(res.masks.data) == 0:
            return self._fallback_mask(img_bgr)

        masks = res.masks.data.cpu().numpy()  # (N, H, W) with H=W=self.input_size
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.ones(len(masks))
        areas = np.array([np.sum(m > self.mask_threshold) for m in masks])
        order = np.argsort(-confs if confs is not None else -areas)
        m = masks[order[0]] > self.mask_threshold

        pt, pl = meta["pad_top"], meta["pad_left"]
        nh, nw = meta["valid_h"], meta["valid_w"]
        submask = m[pt:pt+nh, pl:pl+nw].astype(np.uint8)
        orig_h, orig_w = meta["orig_h"], meta["orig_w"]
        mask_orig = cv2.resize(submask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return (mask_orig > 0).astype(np.uint8)

    def fractal_dimension(self, mask):
        Z = (mask > 0)
        h, w = Z.shape
        max_power = int(np.floor(np.log2(max(2, min(h, w)))))
        sizes = [2**i for i in range(1, max_power + 1)]
        if len(sizes) < 2:
            return 1.0

        counts = []
        scales = []
        for s in sizes:
            H = (h // s) * s
            W = (w // s) * s
            if H == 0 or W == 0:
                continue
            Zc = Z[:H, :W]
            Zc = Zc.reshape(H // s, s, W // s, s)
            blocks = Zc.any(axis=(1, 3))
            N = int(np.count_nonzero(blocks))
            if N > 0:
                counts.append(N)
                scales.append(s)

        if len(counts) < 2:
            return 1.0

        counts = np.asarray(counts, dtype=np.float64)
        scales = np.asarray(scales, dtype=np.float64)
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        D = -coeffs[0]
        return float(D)

    def compute_geometry_features(self, img_pil):
        mask = self.get_mask(img_pil)
        h, w = mask.shape
        img_area = float(h * w)
        diag = float(np.hypot(h, w))
        min_dim = float(max(1, min(h, w)))

        area = float(mask.sum())
        area_norm = area / img_area

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros(11, dtype=np.float32)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        bbox_w = float(x2 - x1 + 1)
        bbox_h = float(y2 - y1 + 1)
        bbox_area = bbox_w * bbox_h
        aspect_ratio = bbox_w / (bbox_h + 1e-6)
        bbox_fill_ratio = area / (bbox_area + 1e-6)

        skel = skeletonize(mask > 0).astype(np.uint8)
        skel_len_px = float(skel.sum())
        skel_len_norm = skel_len_px / (diag + 1e-6)

        dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        widths = (dt[skel > 0] * 2.0)
        if widths.size > 0:
            width_mean = float(np.mean(widths))
            width_max = float(np.max(widths))
        else:
            width_mean = 0.0
            width_max = 0.0
        width_mean_norm = width_mean / min_dim
        width_max_norm = width_max / min_dim

        coords = np.column_stack(np.nonzero(skel))
        if coords.shape[0] >= 2:
            cov = np.cov(coords.T)
            eigvals, eigvecs = np.linalg.eig(cov)
            major = eigvecs[:, np.argmax(eigvals)]
            theta = float(np.arctan2(major[0], major[1]))  # [-pi,pi]
            orient_sin = float(np.sin(theta))
            orient_cos = float(np.cos(theta))
        else:
            orient_sin, orient_cos = 0.0, 1.0

        mask_u8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mean_curv_norm, std_curv_norm = 0.0, 0.0
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            pts = cnt[:, 0, :]
            step = max(1, len(pts) // 200)
            pts = pts[::step]
            curvs = []
            for i in range(1, len(pts) - 1):
                v1 = pts[i] - pts[i - 1]
                v2 = pts[i + 1] - pts[i]
                n1 = np.linalg.norm(v1) + 1e-6
                n2 = np.linalg.norm(v2) + 1e-6
                cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                ang = np.arccos(cosang)
                curvs.append(ang)
            if len(curvs) > 0:
                mean_curv_norm = float(np.mean(np.abs(curvs)) / np.pi)
                std_curv_norm = float(np.std(np.abs(curvs)) / np.pi)

        frac_dim = self.fractal_dimension(mask)
        frac_dim_norm = frac_dim / 2.0

        geom_vec = np.array([
            area_norm,
            bbox_fill_ratio,
            aspect_ratio,
            skel_len_norm,
            width_mean_norm,
            width_max_norm,
            orient_sin,
            orient_cos,
            mean_curv_norm,
            std_curv_norm,
            frac_dim_norm
        ], dtype=np.float32)
        return geom_vec

################## MATCHING METRICS ##################
def match_features(query_feat, memory_feats, metric, attention_model=None):
    if isinstance(query_feat, np.ndarray):
        query_feat = torch.tensor(query_feat, dtype=torch.float32)
    if isinstance(memory_feats, np.ndarray):
        memory_feats = torch.tensor(memory_feats, dtype=torch.float32)

    device = query_feat.device if isinstance(query_feat, torch.Tensor) else torch.device("cpu")
    query_feat = query_feat.to(device)
    memory_feats = memory_feats.to(device)

    if metric == "cosine_similarity":
        query_exp = query_feat.view(1, -1).expand(memory_feats.shape[0], -1)
        sims = F.cosine_similarity(memory_feats, query_exp, dim=1)
        sorted_idx = torch.argsort(-sims)
        return sorted_idx.cpu().numpy(), sims.detach().cpu().numpy()

    elif metric == "attention_similarity":
        if attention_model is None:
            raise ValueError("Attention model required for attention_similarity")
        attention_model.eval()
        with torch.no_grad():
            logits = attention_model(query_feat, memory_feats)  # (num_memory,)
            sims = F.softmax(logits, dim=-1)
            sorted_idx = torch.argsort(-sims)
            return sorted_idx.cpu().numpy(), sims.detach().cpu().numpy()

    else:
        raise ValueError(f"Unsupported metric: {metric}")

################## TRAIN ATTENTION SIMILARITY (CE or TRIPLET) ##################

def train_attention_model(
    train_memory, train_augmented_results, memory_feature_key,
    valid_memory=None, valid_augmented_results=None,
    epochs=50, lr=1e-4, device="cuda", patience=5, min_delta=1e-4,
    loss_type="ce", margin=0.2, neg_strategy="random", hard_negatives_k=50,
    warmup_epochs=20
):
    """
    Train AttentionSimilarity with optional two-stage regime:
      - loss_type='ce': CE only
      - loss_type='triplet': Triplet only, neg_strategy in {'random','semi-hard','hard','topk_stochastic'}
      - loss_type='triplet_schedule': Triplet with 'semi-hard' negatives (50%), topk_stochastic (30%), hard (20%)

    Early stopping is applied per stage (when validation pairs are available).
    """

    def build_pairs(memory, augmented_results, feature_key, device_):
        if memory is None or augmented_results is None:
            return []
        memory_object_ids = [m["object_id"] for m in memory]
        mem_feats = torch.stack(
            [torch.tensor(m[feature_key], dtype=torch.float32) for m in memory]
        ).to(device_)
        pairs_local = []
        for r in augmented_results:
            if "det_feat" not in r or r.get("image_object_id") is None:
                continue
            aug_feat = torch.tensor(r["det_feat"], dtype=torch.float32).to(device_)
            gt_oid = f"{r['image'].replace('.jpg','')}_{r['image_object_id']}"
            if gt_oid not in memory_object_ids:
                continue
            gt_idx = memory_object_ids.index(gt_oid)
            pairs_local.append((aug_feat, mem_feats, gt_idx))
        return pairs_local

    feature_dim = len(train_memory[0][memory_feature_key])
    model = AttentionSimilarity(feature_dim).to(device)

    pairs_train = build_pairs(train_memory, train_augmented_results, memory_feature_key, device)
    pairs_val = build_pairs(valid_memory, valid_augmented_results, memory_feature_key, device)

    n_train, n_val = len(pairs_train), len(pairs_val)
    print(f"Attention training ({loss_type}) on {n_train} train pairs | {n_val} val pairs using '{memory_feature_key}'")
    if n_train == 0:
        print("WARNING: No training pairs. Model stays untrained.")
        return model
    if n_val == 0:
        print("WARNING: No validation pairs provided. Early stopping will use train loss (no val).")

    # Helper: run one training stage
    def run_stage(stage_name, stage_epochs, stage_loss_type, optimizer, margin_local=margin, neg_strategy_local=neg_strategy):
        if stage_epochs <= 0:
            return None, float("inf")

        if stage_loss_type == "ce":
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            loss_fn = nn.TripletMarginLoss(margin=margin_local, p=2, reduction='mean')

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(stage_epochs):
            # ---- Train ----
            model.train()
            random.shuffle(pairs_train)
            total_train_loss = 0.0
            for aug_feat, mem_feats, gt_idx in pairs_train:
                if stage_loss_type == "ce":
                    logits = model(aug_feat, mem_feats).unsqueeze(0)  # (1, num_memory)
                    target = torch.tensor([gt_idx], dtype=torch.long, device=device)
                    loss = loss_fn(logits, target)
                else:  # triplet
                    Q_all = model.encode_query(aug_feat.unsqueeze(0))        # (1, H)
                    K_all = model.encode_keys(mem_feats)                     # (N, H)
                    pos_emb = K_all[gt_idx:gt_idx+1]                         # (1, H)
                    if K_all.shape[0] <= 1:
                        continue  # cannot form a triplet

                    # negative selection
                    if neg_strategy_local == "random":
                        neg_idx = gt_idx
                        while neg_idx == gt_idx and K_all.shape[0] > 1:
                            neg_idx = random.randint(0, K_all.shape[0] - 1)

                    elif neg_strategy_local == "semi-hard":
                        with torch.no_grad():
                            dists = torch.cdist(Q_all, K_all, p=2).squeeze(0)  # (N,)
                            d_pos = dists[gt_idx]
                            mask = torch.ones_like(dists, dtype=torch.bool)
                            mask[gt_idx] = False
                            cand_mask = (dists > d_pos) & (dists < d_pos + margin_local) & mask
                            if cand_mask.any():
                                cand_idxs = torch.where(cand_mask)[0]
                                local_argmin = torch.argmin(dists[cand_idxs])
                                neg_idx = cand_idxs[local_argmin].item()
                            else:
                                neg_pool = torch.where(mask)[0]
                                neg_idx = neg_pool[torch.argmin(dists[neg_pool])].item()

                    elif neg_strategy_local == "topk_stochastic":
                        # Top-k hardest by attention score, then random sample among them
                        with torch.no_grad():
                            logits_ = (Q_all @ K_all.T) / model.temperature  # (1, N)
                            scores = logits_.squeeze(0)                      # (N,)
                            sorted_idx = torch.argsort(scores, descending=True)
                            # exclude the positive
                            sorted_idx = sorted_idx[sorted_idx != gt_idx]
                            k = min(hard_negatives_k, sorted_idx.numel())
                            if k > 0:
                                cand = sorted_idx[:k]
                                neg_idx = int(cand[random.randrange(k)].item())  # uniform random among top-k
                            else:
                                # fallback: random among non-gt
                                neg_pool = torch.arange(K_all.shape[0], device=K_all.device)
                                neg_pool = neg_pool[neg_pool != gt_idx]
                                neg_idx = int(neg_pool[random.randrange(neg_pool.numel())].item())

                    else:  # hard
                        with torch.no_grad():
                            logits_ = (Q_all @ K_all.T) / model.temperature
                            sorted_idx = torch.argsort(logits_.squeeze(0), descending=True)
                            neg_idx = gt_idx
                            for cand in sorted_idx[:hard_negatives_k]:
                                ci = int(cand.item())
                                if ci != gt_idx:
                                    neg_idx = ci
                                    break

                    neg_emb = K_all[neg_idx:neg_idx+1]
                    loss = loss_fn(Q_all, pos_emb, neg_emb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, len(pairs_train))

            # ---- Validation ----
            model.eval()
            total_val_loss, val_count = 0.0, 0
            with torch.no_grad():
                for aug_feat, mem_feats, gt_idx in pairs_val:
                    if stage_loss_type == "ce":
                        logits = model(aug_feat, mem_feats).unsqueeze(0)
                        target = torch.tensor([gt_idx], dtype=torch.long, device=device)
                        loss = loss_fn(logits, target)
                    else:
                        Q_all = model.encode_query(aug_feat.unsqueeze(0))
                        K_all = model.encode_keys(mem_feats)
                        pos_emb = K_all[gt_idx:gt_idx+1]
                        if K_all.shape[0] <= 1:
                            continue

                        if neg_strategy_local == "random":
                            neg_idx = gt_idx
                            while neg_idx == gt_idx and K_all.shape[0] > 1:
                                neg_idx = random.randint(0, K_all.shape[0] - 1)

                        elif neg_strategy_local == "semi-hard":
                            dists = torch.cdist(Q_all, K_all, p=2).squeeze(0)  # (N,)
                            d_pos = dists[gt_idx]
                            mask = torch.ones_like(dists, dtype=torch.bool)
                            mask[gt_idx] = False
                            cand_mask = (dists > d_pos) & (dists < d_pos + margin_local) & mask
                            if cand_mask.any():
                                cand_idxs = torch.where(cand_mask)[0]
                                local_argmin = torch.argmin(dists[cand_idxs])
                                neg_idx = cand_idxs[local_argmin].item()
                            else:
                                neg_pool = torch.where(mask)[0]
                                neg_idx = neg_pool[torch.argmin(dists[neg_pool])].item()

                        elif neg_strategy_local == "topk_stochastic":
                            logits_ = (Q_all @ K_all.T) / model.temperature
                            scores = logits_.squeeze(0)
                            sorted_idx = torch.argsort(scores, descending=True)
                            sorted_idx = sorted_idx[sorted_idx != gt_idx]
                            k = min(hard_negatives_k, sorted_idx.numel())
                            if k > 0:
                                cand = sorted_idx[:k]
                                neg_idx = int(cand[random.randrange(k)].item())
                            else:
                                neg_pool = torch.arange(K_all.shape[0], device=K_all.device)
                                neg_pool = neg_pool[neg_pool != gt_idx]
                                neg_idx = int(neg_pool[random.randrange(neg_pool.numel())].item())

                        else:  # hard
                            logits_ = (Q_all @ K_all.T) / model.temperature
                            sorted_idx = torch.argsort(logits_.squeeze(0), descending=True)
                            neg_idx = gt_idx
                            for cand in sorted_idx[:hard_negatives_k]:
                                ci = int(cand.item())
                                if ci != gt_idx:
                                    neg_idx = ci
                                    break

                        neg_emb = K_all[neg_idx:neg_idx+1]
                        loss = loss_fn(Q_all, pos_emb, neg_emb)

                    total_val_loss += loss.item()
                    val_count += 1

            avg_val_loss = (total_val_loss / max(1, val_count)) if n_val > 0 else avg_train_loss
            print(f"{stage_name} Epoch {epoch+1}/{stage_epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            metric_for_es = avg_val_loss if n_val > 0 else avg_train_loss
            if metric_for_es + min_delta < best_val_loss:
                best_val_loss = metric_for_es
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping {stage_name} at epoch {epoch+1} (best {'val' if n_val > 0 else 'train'} loss={best_val_loss:.4f})")
                    break

        return best_state, best_val_loss

    # === Branches ===
    if loss_type in ("ce", "triplet"):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        stage_name = "CE" if loss_type == "ce" else f"Triplet({neg_strategy})"
        best_state, _ = run_stage(stage_name, epochs, loss_type, optimizer, margin_local=margin, neg_strategy_local=neg_strategy)
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    elif loss_type == "triplet_schedule":
        """
        Four‐stage training:
          0) CE warmup       = 10% of total epochs
          1) Triplet semi-hard = 40% of remaining
          2) Triplet top-k stochastic = 30% of remaining
          3) Triplet hard     = 20% of remaining
        """
        # 0) CE warmup
        warmup0 = max(1, int(epochs * 0.05))
        rest    = epochs - warmup0
        #semi_e  = int(rest * 0.2)
        topk_e  = int(rest * 0.8)
        hard_e  = rest - topk_e

        print(f"[Schedule] CE warmup {warmup0} → "
              f"TopKStochastic {topk_e} → "
              f"Hard {hard_e}")

        # Stage 0: CE warmup
        opt0 = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        best0, _ = run_stage("Warmup-CE", warmup0, "ce", opt0)
        if best0 is not None:
            model.load_state_dict(best0)

        # Stage 1: Triplet semi-hard
        #opt1 = torch.optim.Adam(model.parameters(), lr=lr * 0.5, weight_decay=1e-4)
        #best1, _ = run_stage(
        #    "Triplet-SemiHard", semi_e, "triplet", opt1,
        #    margin_local=margin, neg_strategy_local="semi-hard"
        #)
        #if best1 is not None:
        #    model.load_state_dict(best1)

        # Stage 2: Triplet top-k stochastic
        opt2 = torch.optim.Adam(model.parameters(), lr=lr * 0.5, weight_decay=1e-4)
        best2, _ = run_stage(
            "Triplet-TopKStochastic", topk_e, "triplet", opt2,
            margin_local=margin, neg_strategy_local="topk_stochastic"
        )
        if best2 is not None:
            model.load_state_dict(best2)

        # Stage 3: Triplet hard
        opt3 = torch.optim.Adam(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
        best3, _ = run_stage(
            "Triplet-Hard", hard_e, "triplet", opt3,
            margin_local=margin, neg_strategy_local="hard"
        )
        if best3 is not None:
            model.load_state_dict(best3)

        return model

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

################## MEMORY BUILDING ##################
def build_feature_memory(original_mapping, extractor, geometry_extractor=None):
    memory = []
    for object_id, data in original_mapping.items():
        img_path = data["image_file"]
        with Image.open(img_path).convert("RGB") as img:
            base_feat = extractor.extract(img)  # already L2-normalized
            entry = {"feature": base_feat, "object_id": object_id, "image_file": img_path}
            if geometry_extractor is not None:
                geom_feat = geometry_extractor.compute_geometry_features(img)
                geom_feat = l2_normalize_np(geom_feat)  # L2-normalize geometry features
                entry["geom_feature"] = geom_feat.astype(np.float32)
                concat = np.concatenate([base_feat.astype(np.float32), geom_feat.astype(np.float32)], axis=0)
                concat = l2_normalize_np(concat)  # L2-normalize concatenated vector
                entry["concat_feature"] = concat
            else:
                entry["geom_feature"] = None
                entry["concat_feature"] = None
            memory.append(entry)
    return memory

################## PLOT- UND VISUALISIERUNGSFUNKTIONEN ######################
def normalize(X):
    X = np.asarray(X)
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm[X_norm == 0] = 1
    return X / X_norm

def compute_attention_distance_matrix(X, attention_model, device=None):
    """
    Build a symmetric distance matrix from attention logits.
    sims_ij = (Q_i·K_j + Q_j·K_i)/2 (with normalized embeddings) then min-max normalize and convert to distance: d = 1 - sims_norm.
    """
    if attention_model is None:
        raise ValueError("attention_model is required for attention distance")
    attention_model.eval()
    if device is None:
        device = next(attention_model.parameters()).device
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        Q = attention_model.encode_query(X_t)  # (N, H), normalized
        K = attention_model.encode_keys(X_t)   # (N, H), normalized
        logits_qk = (Q @ K.T) / attention_model.temperature  # (N, N)
        logits_kq = (K @ Q.T) / attention_model.temperature  # (N, N)
        sims = (logits_qk + logits_kq) / 2.0
        sims = sims.detach().cpu().numpy()
    # Min-max normalize similarities to [0,1]
    s_min = np.min(sims)
    s_max = np.max(sims)
    if s_max - s_min < 1e-12:
        sims_norm = np.zeros_like(sims)
    else:
        sims_norm = (sims - s_min) / (s_max - s_min)
    dists = 1.0 - sims_norm
    np.fill_diagonal(dists, 0.0)
    # Ensure symmetry and non-negativity
    dists = (dists + dists.T) / 2.0
    dists[dists < 0] = 0
    return dists

def compute_dist_matrix(X, metric, attention_model=None, device=None):
    if metric == "cosine_similarity":
        return cosine_distances(X, X)
    elif metric == "euclidean_distance":
        return euclidean_distances(X, X)
    elif metric == "manhattan":
        return manhattan_distances(X, X)
    elif metric == "attention_similarity":
        return compute_attention_distance_matrix(X, attention_model, device=device)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def plot_success_and_failure_examples(results, memory, augmented_dir, original_dir, output_dir):
    successes = [r for r in results if r["is_success"]]
    failures = [r for r in results if not r["is_success"]]
    success_examples = random.sample(successes, min(4, len(successes)))
    failure_examples = random.sample(failures, min(4, len(failures)))

    print("Plotting success examples...")
    for idx, example in enumerate(success_examples):
        plot_single_example(example, memory, augmented_dir, original_dir, output_dir, f"success_{idx + 1}")

    print("Plotting failure examples...")
    for idx, example in enumerate(failure_examples):
        plot_single_example(example, memory, augmented_dir, original_dir, output_dir, f"failure_{idx + 1}")

def plot_single_example(example, memory, augmented_dir, original_dir, output_dir, plot_name):
    img_file = example["image"]
    det_bbox = example["detection_bbox"]
    best_object_id = example["best_object_id"]
    is_success = example["is_success"]

    aug_img_path = os.path.join(augmented_dir, 'images', img_file)
    original_img_path = os.path.join(original_dir, 'images', img_file)

    if not os.path.exists(aug_img_path) or not os.path.exists(original_img_path):
        print(f"Image not found for {img_file}, skipping...")
        return

    aug_img = Image.open(aug_img_path).convert("RGB")
    original_img = Image.open(original_img_path).convert("RGB")

    aug_img_draw = ImageDraw.Draw(aug_img)
    aug_img_draw.rectangle(det_bbox, outline="red", width=3)

    predicted_crop_path = None
    for mem in memory:
        if mem["object_id"] == best_object_id:
            predicted_crop_path = mem["image_file"]
            break

    if not predicted_crop_path:
        print(f"Predicted crop not found for object ID: {best_object_id}, skipping...")
        return

    predicted_crop = Image.open(predicted_crop_path).convert("RGB")
    detected_crop = aug_img.crop(det_bbox)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"{'Success' if is_success else 'Failure'} Example", fontsize=16, color=("green" if is_success else "red"))

    axes[0].imshow(original_img); axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(aug_img); axes[1].set_title("Augmented Image\n(Detection in Red)"); axes[1].axis("off")
    axes[2].imshow(detected_crop); axes[2].set_title("Detected Crop"); axes[2].axis("off")
    axes[3].imshow(predicted_crop); axes[3].set_title("Predicted Crop"); axes[3].axis("off")

    plot_path = os.path.join(output_dir, f"{plot_name}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Example plot saved: {plot_path}")

def plot_tsne_memory_and_augmented(
    memory, augmented_results, output_dir, metric="cosine_similarity", max_imgs=1000, feature_key="feature",
    attention_model=None, device=None
):
    memory_feats, memory_imgs, memory_obj_ids = [], [], []
    for r in memory:
        if feature_key in r and r[feature_key] is not None and "image_file" in r:
            memory_feats.append(r[feature_key])
            memory_imgs.append(r.get("image_file", None))
            memory_obj_ids.append(r.get("object_id", None))
    aug_feats, aug_imgs, aug_obj_ids = [], [], []
    for r in augmented_results:
        if "det_feat" in r and "det_crop_img" in r:
            aug_feats.append(r["det_feat"])
            aug_imgs.append(r["det_crop_img"])
            if r.get("image") is not None and r.get("image_object_id") is not None:
                aug_oid_full = f"{r['image'].replace('.jpg','')}_{r['image_object_id']}"
            else:
                aug_oid_full = None
            aug_obj_ids.append(aug_oid_full)

    if len(memory_feats) == 0 or len(aug_feats) == 0:
        print("Not enough features for t-SNE plot.")
        return None, None

    if len(memory_feats) > max_imgs:
        idxs = np.random.choice(len(memory_feats), max_imgs, replace=False)
        memory_feats = [memory_feats[i] for i in idxs]
        memory_imgs = [memory_imgs[i] for i in idxs]
        memory_obj_ids = [memory_obj_ids[i] for i in idxs]
    if len(aug_feats) > max_imgs:
        idxs = np.random.choice(len(aug_feats), max_imgs, replace=False)
        aug_feats = [aug_feats[i] for i in idxs]
        aug_imgs = [aug_imgs[i] for i in idxs]
        aug_obj_ids = [aug_obj_ids[i] for i in idxs]

    all_feats = np.vstack([memory_feats, aug_feats])

    # Normalize unless attention metric (we use attention model projection there)
    if metric != "attention_similarity":
        feats_for_dist = normalize(all_feats)
    else:
        feats_for_dist = all_feats

    dist_matrix = compute_dist_matrix(feats_for_dist, metric, attention_model=attention_model, device=device)

    n_points = dist_matrix.shape[0]
    # Perplexity must be < n_points; choose adaptively
    perplexity = min(30, max(5, (n_points - 1) // 3)) if n_points > 5 else max(2, n_points // 2)
    tsne = TSNE(n_components=2, metric='precomputed', random_state=42, perplexity=perplexity, init="random")
    X_embedded = tsne.fit_transform(dist_matrix)
    X_mem = X_embedded[:len(memory_feats)]
    X_aug = X_embedded[len(memory_feats):]

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_title(f't-SNE: Blue=Memory, Orange=Augmented\nDistance: {metric} | {feature_key}', fontsize=17)
    ax.set_xticks([]); ax.set_yticks([])

    obj_id_to_mem_idx = {oid: i for i, oid in enumerate(memory_obj_ids)}
    for j, aug_oid in enumerate(aug_obj_ids):
        if aug_oid in obj_id_to_mem_idx:
            i = obj_id_to_mem_idx[aug_oid]
            ax.plot(
                [X_mem[i,0], X_aug[j,0]],
                [X_mem[i,1], X_aug[j,1]],
                linestyle=':', color='black', linewidth=2, alpha=0.7, zorder=2
            )

    ax.scatter(X_mem[:,0], X_mem[:,1], c='royalblue', marker='o', s=90,
               label=f'Memory ({feature_key})', alpha=0.8, edgecolor='k', zorder=3)
    ax.scatter(X_aug[:,0], X_aug[:,1], c='orange', marker='*', s=140,
               label='Augmented', alpha=0.8, edgecolor='k', zorder=4)

    ax.legend(fontsize=14, loc='best')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"tsne_memory_and_augmented_{metric}_{feature_key}.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"t-SNE plot saved at: {save_path}")
    return X_mem, X_aug

def plot_tsne_memory_thumbnails_at_coords(X_mem, memory, output_dir, metric="cosine_similarity", feature_key="feature", thumb_size=48, zoom=1.0, keep_aspect_ratio=True):
    if X_mem is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    imgfiles = [r["image_file"] for r in memory if feature_key in r and r[feature_key] is not None and "image_file" in r]
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_title(f't-SNE: Memory Thumbnails ({metric} | {feature_key})', fontsize=17)
    ax.set_xticks([]); ax.set_yticks([])
    count = 0
    for xy, img_path in zip(X_mem, imgfiles):
        try:
            img = Image.open(img_path).convert("RGB")
            if keep_aspect_ratio:
                img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
            else:
                img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
            im = OffsetImage(np.array(img), zoom=zoom)
            ab = AnnotationBbox(im, xy, frameon=False, pad=0.01)
            ax.add_artist(ab)
            count += 1
        except Exception as e:
            print(f"Could not load {img_path}: {e}")
    if count > 0:
        min_x, max_x = X_mem[:,0].min(), X_mem[:,0].max()
        min_y, max_y = X_mem[:,1].min(), X_mem[:,1].max()
        pad_x = (max_x - min_x) * 0.05
        pad_y = (max_y - min_y) * 0.05
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)
        plt.tight_layout()
        aspect_str = "aspect" if keep_aspect_ratio else "square"
        save_path = os.path.join(output_dir, f"tsne_memory_thumbnails_{metric}_{feature_key}_{aspect_str}.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"t-SNE thumbnail plot saved at: {save_path}")
    else:
        print("No valid images to plot, nothing saved.")

def plot_topk_example(
    example, memory, augmented_dir, original_dir, output_dir, plot_name,
    top_k=3, metric="cosine_similarity", feature_key="feature", attention_model=None
):
    img_file = example["image"]
    det_bbox = example["detection_bbox"]
    det_feat = example["det_feat"]

    aug_img_path = os.path.join(augmented_dir, 'images', img_file)
    original_img_path = os.path.join(original_dir, 'images', img_file)
    if not (os.path.exists(aug_img_path) and os.path.exists(original_img_path)):
        print(f"Image not found for {img_file}, skipping...")
        return

    aug_img = Image.open(aug_img_path).convert("RGB")
    original_img = Image.open(original_img_path).convert("RGB")
    detected_crop = aug_img.crop(det_bbox)

    feature_matrix = np.array([m[feature_key] for m in memory], dtype=np.float32)

    if metric in ["cosine_similarity", "attention_similarity"]:
        sorted_idx, sims = match_features(det_feat.astype(np.float32), feature_matrix, metric, attention_model)
    elif metric == "euclidean_distance":
        dists = euclidean_distances([det_feat], feature_matrix)[0]
        sorted_idx = np.argsort(dists)
        sims = -dists
    elif metric == "manhattan":
        dists = np.array([cityblock(det_feat, m_feat) for m_feat in feature_matrix])
        sorted_idx = np.argsort(dists)
        sims = -dists
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    top_matches = [memory[i] for i in sorted_idx[:top_k]]

    correct_object_id = example.get("image_object_id", None)
    img_prefix = img_file.replace('.jpg', '')
    gt_crop_path, gt_rank = None, None
    if correct_object_id is not None:
        gt_object_id = f"{img_prefix}_{correct_object_id}"
        memory_object_ids = [m["object_id"] for m in memory]
        if gt_object_id in memory_object_ids:
            gt_idx = memory_object_ids.index(gt_object_id)
            gt_crop_path = memory[gt_idx]["image_file"]
            gt_rank = int(np.where(sorted_idx == gt_idx)[0][0]) + 1

    worst_idx = sorted_idx[-1]
    worst_crop_path = memory[worst_idx]["image_file"]
    show_gt = (gt_crop_path is not None and gt_rank is not None and gt_rank > top_k)
    n_cols = 3 + top_k + (1 if show_gt else 0) + 1

    title_color = 'green' if (gt_rank == 1 if gt_rank is not None else False) else ('red' if gt_rank is not None else 'black')
    title_text = f"Top-k ({feature_key}, {metric}) | GT Rank: {gt_rank}" if gt_rank is not None else f"Top-k ({feature_key}, {metric}) (GT not found)"

    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    fig.suptitle(title_text, fontsize=22, color=title_color)

    col = 0
    axes[col].imshow(original_img); axes[col].set_title("Original"); axes[col].axis("off"); col += 1
    aug_img_draw = aug_img.copy()
    draw = ImageDraw.Draw(aug_img_draw)
    draw.rectangle(det_bbox, outline="red", width=3)
    axes[col].imshow(aug_img_draw); axes[col].set_title("Augmented (Det)"); axes[col].axis("off"); col += 1
    axes[col].imshow(detected_crop); axes[col].set_title("Detected Crop"); axes[col].axis("off"); col += 1

    for k in range(top_k):
        try:
            crop_img = Image.open(top_matches[k]["image_file"]).convert("RGB")
        except Exception:
            crop_img = Image.new("RGB", (32, 32), (255, 0, 0))
        axes[col].imshow(crop_img)
        if gt_crop_path and top_matches[k]["image_file"] == gt_crop_path:
            color = 'green' if k == 0 else 'red'
            axes[col].set_title(f"Rank {k + 1} (GT)", color=color)
            for spine in axes[col].spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(3)
        else:
            axes[col].set_title(f"Rank {k + 1}")
        axes[col].axis("off"); col += 1

    if show_gt:
        try:
            gt_crop_img = Image.open(gt_crop_path).convert("RGB")
            axes[col].imshow(gt_crop_img)
            color = 'green' if gt_rank == 1 else 'red'
            axes[col].set_title(f"GT Crop\n(Rank {gt_rank})", color=color)
            for spine in axes[col].spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(3)
        except Exception:
            axes[col].imshow(np.zeros((32, 32, 3), dtype=np.uint8))
        axes[col].axis("off"); col += 1

    try:
        worst_crop_img = Image.open(worst_crop_path).convert("RGB")
        axes[col].imshow(worst_crop_img)
    except Exception:
        axes[col].imshow(np.zeros((32, 32, 3), dtype=np.uint8))
    axes[col].set_title(f"Worst\n(Rank {len(memory)})"); axes[col].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{plot_name}_top{top_k}_{feature_key}_{metric}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Top-K plot saved: {save_path}")

def compute_cmc(gt_ranks, num_gallery, max_rank_plot=50):
    """
    gt_ranks: list of integer ranks (1-based) for valid queries; ignore None.
    num_gallery: total number of gallery items (size of memory).
    max_rank_plot: limit the plotted rank.
    """
    ranks = [r for r in gt_ranks if r is not None and r >= 1]
    if len(ranks) == 0:
        return np.zeros(min(max_rank_plot, num_gallery), dtype=np.float32)
    max_r = min(max_rank_plot, num_gallery)
    cmc = np.zeros(max_r, dtype=np.float32)
    ranks_np = np.array(ranks, dtype=np.int32)
    for k in range(1, max_r + 1):
        cmc[k - 1] = np.mean(ranks_np <= k)
    return cmc

def plot_and_save_cmc(cmc, output_dir, filename_prefix, title="CMC Curve"):
    os.makedirs(output_dir, exist_ok=True)
    x = np.arange(1, len(cmc) + 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, cmc * 100.0, marker='o', color='tab:blue', label='CMC')
    ax.set_xlabel("Rank")
    ax.set_ylabel("Matching Rate (%)")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(1, len(cmc))
    ax.set_ylim(0, 100)
    ax.legend()
    save_path = os.path.join(output_dir, f"{filename_prefix}_cmc.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    # Save CMC values
    cmc_csv = os.path.join(output_dir, f"{filename_prefix}_cmc.csv")
    pd.DataFrame({"rank": x, "cmc": cmc}).to_csv(cmc_csv, index=False)
    print(f"CMC saved: {save_path} and {cmc_csv}")
    # Return common CMC points
    def pick(idx):
        return cmc[idx - 1] if len(cmc) >= idx else np.nan
    return pick(1), pick(5), pick(10)

def compute_per_object_reid_stats(results, present_object_ids):
    """
    Build per-object stats: queries, top1_rate, mean_rank, median_rank.
    """
    per_obj = {}
    for r in results:
        gt_oid = r.get("gt_object_id", None)
        gt_rank = r.get("gt_rank", None)
        if gt_oid is None or gt_oid not in present_object_ids:
            continue
        if gt_oid not in per_obj:
            per_obj[gt_oid] = {"ranks": []}
        if gt_rank is not None and gt_rank >= 1:
            per_obj[gt_oid]["ranks"].append(gt_rank)
    rows = []
    for oid, v in per_obj.items():
        ranks = v["ranks"]
        if len(ranks) == 0:
            continue
        ranks_np = np.array(ranks)
        rows.append({
            "object_id": oid,
            "num_queries": len(ranks),
            "top1_rate": float(np.mean(ranks_np == 1)),
            "top5_rate": float(np.mean(ranks_np <= 5)),
            "mean_rank": float(np.mean(ranks_np)),
            "median_rank": float(np.median(ranks_np))
        })
    df = pd.DataFrame(rows).sort_values(by=["top1_rate", "num_queries"], ascending=[False, False])
    return df

def plot_topk_examples_for_success_and_failure(
    results, memory, augmented_dir, original_dir, output_dir,
    top_k=3, feature_key="feature", metric="cosine_similarity", attention_model=None
):
    successes = [r for r in results if r["is_success"]]
    failures = [r for r in results if not r["is_success"]]

    success_examples = random.sample(successes, min(4, len(successes)))
    failure_examples = random.sample(failures, min(4, len(failures)))

    print("Plotting Top-K success examples...")
    for idx, example in enumerate(success_examples):
        plot_topk_example(
            example, memory, augmented_dir, original_dir, output_dir,
            f"topk_success_{idx+1}", top_k=top_k,
            metric=metric, feature_key=feature_key, attention_model=attention_model
        )

    print("Plotting Top-K failure examples...")
    for idx, example in enumerate(failure_examples):
        plot_topk_example(
            example, memory, augmented_dir, original_dir, output_dir,
            f"topk_failure_{idx+1}", top_k=top_k,
            metric=metric, feature_key=feature_key, attention_model=attention_model
        )

################## EVALUATION ##################
def evaluate_with_yolo(augmented_dir, yolo_weights, extractor, memory, metric="cosine_similarity",
                       confidence_threshold=0.5, attention_model=None,
                       feature_key="feature", geometry_extractor=None):
    """
    Evaluate detection and Re-ID matching on chosen feature representation.

    feature_key: 'feature', 'concat_feature', or 'geom_feature' (Memory)
    det_feat built accordingly:
      - 'feature' -> DINO only (already L2-normalized)
      - 'concat_feature' -> concat(L2-normalized DINO, L2-normalized geometry), then L2-normalize again
      - 'geom_feature' -> L2-normalized geometry only
    """
    yolo = YOLO(yolo_weights)
    images_dir = os.path.join(augmented_dir, 'images')
    labels_dir = os.path.join(augmented_dir, 'labels')
    results = []
    total_detections = 0
    total_objects = 0
    top1_success_count = 0
    all_object_ids = [m["object_id"] for m in memory]
    y_true, y_scores = [], []

    for label_file in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            total_objects += len(f.readlines())

    for img_file in os.listdir(images_dir):
        if not img_file.endswith('.jpg'):
            continue
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt'))
        if not os.path.exists(label_path):
            continue
        image = Image.open(img_path).convert("RGB")
        detections = yolo(img_path, conf=confidence_threshold)[0].boxes.xyxy.cpu().numpy()
        with open(label_path, 'r') as f:
            labels = f.readlines()

        for det_bbox in detections:
            total_detections += 1
            crop = image.crop(det_bbox[:4])

            # Build detection feature according to feature_key
            base_feat = extractor.extract(crop)  # already L2-normalized
            if feature_key == 'concat_feature':
                if geometry_extractor is None:
                    raise ValueError("geometry_extractor required when feature_key='concat_feature'")
                geom_feat = geometry_extractor.compute_geometry_features(crop)
                geom_feat = l2_normalize_np(geom_feat)
                det_feat = np.concatenate([base_feat.astype(np.float32), geom_feat.astype(np.float32)], axis=0)
                det_feat = l2_normalize_np(det_feat)
            elif feature_key == 'geom_feature':
                if geometry_extractor is None:
                    raise ValueError("geometry_extractor required when feature_key='geom_feature'")
                geom_feat = geometry_extractor.compute_geometry_features(crop)
                det_feat = l2_normalize_np(geom_feat).astype(np.float32)
            else:
                det_feat = base_feat.astype(np.float32)

            feature_matrix = torch.stack([
                torch.tensor(m[feature_key], dtype=torch.float32).to(extractor.device)
                for m in memory
            ])
            sorted_idx, sims = match_features(det_feat, feature_matrix, metric, attention_model)

            best_match = memory[sorted_idx[0]]
            best_object_id = best_match["object_id"]

            matched_object_number, matched_label_bbox, max_iou = None, None, 0
            for i, label in enumerate(labels):
                parts = label.strip().split()
                polygon_points = np.array(list(map(float, parts[1:]))).reshape(-1, 2)
                width, height = image.size
                polygon_points[:, 0] *= width
                polygon_points[:, 1] *= height
                label_bbox = polygon_to_bbox(polygon_points)
                iou = compare_bboxes(det_bbox, label_bbox)
                if iou > max_iou:
                    max_iou = iou
                    matched_object_number = f"object{i}"
                    matched_label_bbox = label_bbox

            reconstructed_object_id = f"{img_file.split('.jpg')[0]}_{matched_object_number}"

            gt_idx = None
            gt_rank = None
            gt = np.zeros(len(all_object_ids))
            if reconstructed_object_id in all_object_ids:
                gt_idx = all_object_ids.index(reconstructed_object_id)
                gt[gt_idx] = 1
                pos = np.flatnonzero(sorted_idx == gt_idx)
                if pos.size > 0:
                    gt_rank = int(pos[0]) + 1

            is_success = (gt_rank == 1)

            y_true.append(gt)
            y_scores.append(sims)

            results.append({
                "image": img_file,
                "image_object_id": matched_object_number,
                "detection_bbox": det_bbox.tolist(),
                "label_bbox": matched_label_bbox,
                "iou": max_iou,
                "best_object_id": best_object_id,
                "is_success": is_success,
                "gt_rank": gt_rank,
                "gt_object_id": reconstructed_object_id,
                "det_feat": det_feat,
                "det_crop_img": np.array(crop)
            })
            if is_success:
                top1_success_count += 1

    success_rate_detections = (top1_success_count / total_detections) * 100 if total_detections > 0 else 0
    success_rate_objects = (top1_success_count / total_objects) * 100 if total_objects > 0 else 0

    y_true_np = np.array(y_true)
    y_scores_np = np.array(y_scores)
    ap_per_class_standard = []
    for i in range(y_true_np.shape[1]):
        if np.sum(y_true_np[:, i]) > 0:
            ap = average_precision_score(y_true_np[:, i], y_scores_np[:, i])
            ap_per_class_standard.append(ap)
    mAP_standard = np.mean(ap_per_class_standard) if ap_per_class_standard else 0

    present_object_ids = set()
    labels_dir_local = os.path.join(augmented_dir, 'labels')
    for label_file in os.listdir(labels_dir_local):
        with open(os.path.join(labels_dir_local, label_file), 'r') as f:
            labels = f.readlines()
            for i, label in enumerate(labels):
                present_object_ids.add(f"{label_file.replace('.txt','')}_object{i}")

    ap_per_class_fair = []
    for i, object_id in enumerate(all_object_ids):
        if object_id in present_object_ids:
            if np.sum(y_true_np[:, i]) > 0:
                ap = average_precision_score(y_true_np[:, i], y_scores_np[:, i])
            else:
                ap = 0
            ap_per_class_fair.append(ap)
    mAP_fair = np.mean(ap_per_class_fair) if ap_per_class_fair else 0

    return {
        "success_rate_detections": success_rate_detections,
        "success_rate_objects": success_rate_objects,
        "mAP_standard": mAP_standard,
        "mAP_fair": mAP_fair,
        "results": results,
        "total_detections": total_detections,
        "total_objects": total_objects,
        "present_object_ids": present_object_ids,
        "all_object_ids": all_object_ids
    }

################## MAIN ##################
# python
if __name__ == "__main__":
    # Paths
    train_original_mapping_file = '/workspace/runs/reidentification_results/runs/train_cropped_dataset/object_mapping.json'
    augmented_train_dir = '/workspace/runs/reidentification_results/runs/augmented_train_dataset'
    valid_original_mapping_file = '/workspace/runs/reidentification_results/runs/valid_cropped_dataset/object_mapping.json'
    augmented_valid_dir = '/workspace/runs/reidentification_results/runs/augmented_valid_dataset'
    test_original_mapping_file = '/workspace/runs/reidentification_results/runs/test_cropped_dataset/object_mapping.json'
    augmented_test_dir = '/workspace/runs/reidentification_results/runs/augmented_test_dataset'
    original_test_dir = '/workspace/test'
    yolo_weights_list = ['/workspace/runs/reidentification_results/best-seg_baseline.pt']          # detection weights
    yolo_seg_weights = '/workspace/runs/reidentification_results/best-seg_baseline.pt'         # segmentation weights for geometry features
    output_dir = '/workspace/runs/reidentification_results/evaluation_results_mAP_baseline_run_1'
    backbones = ['dinov2']
    confidence_thresholds = [0.4]

    # Loss variants for Attention training
    attention_loss_types = ['ce', 'triplet', 'triplet_schedule']  # added ce_then_triplet warmup regime
    #attention_loss_types = ['ce', 'triplet_schedule']  # added ce_then_triplet warmup regime
    triplet_neg_strategies = ['random', 'semi-hard', 'hard', 'topk_stochastic']  # for pure triplet branch
    #triplet_neg_strategies = ['topk_stochastic']  # for pure triplet branch
    triplet_margin = 0.5
    warmup_epochs = 20  # CE warmup epochs for ce_then_triplet

    # Feature sets to evaluate
    feature_keys_to_eval = ['feature', 'concat_feature', 'geom_feature']
    #feature_keys_to_eval = ['feature']

    os.makedirs(output_dir, exist_ok=True)

    for yolo_weights in yolo_weights_list:
        # Option 1 (os.path)
        weight_tag = os.path.splitext(os.path.basename(yolo_weights))[0]  # 'best-seg_baseline'
        model_output_dir = os.path.join(output_dir, weight_tag)
        os.makedirs(model_output_dir, exist_ok=True)

        for backbone in backbones:
            # Feature extractor
            extractor = FlexibleFeatureExtractor(backbone=backbone)
            geometry_extractor = GeometryFeatureExtractor(
                seg_weights_path=yolo_seg_weights, input_size=640, conf_threshold=0.1, mask_threshold=0.5
            )

            # Build Train Memory (including geom and concat features, both normalized)
            with open(train_original_mapping_file, 'r') as f:
                train_original_mapping = json.load(f)
            train_memory = build_feature_memory(train_original_mapping, extractor, geometry_extractor=geometry_extractor)

            # Build Valid Memory
            with open(valid_original_mapping_file, 'r') as f:
                valid_original_mapping = json.load(f)
            valid_memory = build_feature_memory(valid_original_mapping, extractor, geometry_extractor=geometry_extractor)

            # Build Test Memory (including geom and concat features)
            with open(test_original_mapping_file, 'r') as f:
                test_original_mapping = json.load(f)
            test_memory = build_feature_memory(test_original_mapping, extractor, geometry_extractor=geometry_extractor)

            # Loop over confidence thresholds
            for confidence_threshold in confidence_thresholds:
                config_output_dir = os.path.join(
                    model_output_dir, f"{backbone}_conf_{confidence_threshold}"
                )
                os.makedirs(config_output_dir, exist_ok=True)

                summary_rows = []

                # Evaluate for all feature sets: 'feature', 'concat_feature', 'geom_feature'
                for feature_key in feature_keys_to_eval:
                    print(f"\n=== Feature Set: {feature_key} ===")

                    # 1) Evaluate TRAIN (cosine) for creating attention-training pairs (det_feat according to feature_key)
                    print("Evaluating YOLO on augmented TRAIN dataset for Attention training...")
                    train_stats_for_attention = evaluate_with_yolo(
                        augmented_train_dir, yolo_weights, extractor, train_memory,
                        metric='cosine_similarity',
                        confidence_threshold=confidence_threshold,
                        attention_model=None,
                        feature_key=feature_key,
                        geometry_extractor=geometry_extractor
                    )

                    # VALID pairs (for attention validation/early stopping)
                    print("Evaluating YOLO on augmented VALID dataset for Attention validation pairs...")
                    valid_stats_for_attention = evaluate_with_yolo(
                        augmented_valid_dir, yolo_weights, extractor, valid_memory,
                        metric='cosine_similarity',
                        confidence_threshold=confidence_threshold,
                        attention_model=None,
                        feature_key=feature_key,
                        geometry_extractor=geometry_extractor
                    )

                    # 2) Test evaluation with COSINE_SIMILARITY baseline
                    print("Evaluating YOLO on augmented TEST dataset (cosine)...")
                    test_stats_cosine = evaluate_with_yolo(
                        augmented_test_dir, yolo_weights, extractor, test_memory,
                        metric="cosine_similarity",
                        confidence_threshold=confidence_threshold,
                        attention_model=None,
                        feature_key=feature_key,
                        geometry_extractor=geometry_extractor
                    )
                    cos_dir = os.path.join(config_output_dir, f"{feature_key}_cosine")
                    os.makedirs(cos_dir, exist_ok=True)
                    results_df_cos = pd.DataFrame(test_stats_cosine["results"])
                    results_df_cos.to_csv(os.path.join(cos_dir, "results.csv"), index=False)
                    with open(os.path.join(cos_dir, "results.md"), "w") as f:
                        f.write(results_df_cos.to_markdown(index=False))
                    # Plots
                    plot_success_and_failure_examples(
                        test_stats_cosine["results"], test_memory,
                        augmented_test_dir, original_test_dir, cos_dir
                    )
                    plot_topk_examples_for_success_and_failure(
                        test_stats_cosine["results"], test_memory,
                        augmented_test_dir, original_test_dir,
                        cos_dir, top_k=3, feature_key=feature_key,
                        metric="cosine_similarity", attention_model=None
                    )
                    # t-SNE using cosine distance
                    X_mem, X_aug = plot_tsne_memory_and_augmented(
                        test_memory, test_stats_cosine["results"], cos_dir, metric='cosine_similarity', feature_key=feature_key,
                        attention_model=None, device=extractor.device
                    )
                    plot_tsne_memory_thumbnails_at_coords(
                        X_mem, test_memory, cos_dir, metric='cosine_similarity', feature_key=feature_key, keep_aspect_ratio=True
                    )
                    # CMC computation and plot
                    gt_ranks_cos = [r.get("gt_rank", None) for r in test_stats_cosine["results"]]
                    cmc_cos = compute_cmc(gt_ranks_cos, num_gallery=len(test_memory), max_rank_plot=50)
                    r1, r5, r10 = plot_and_save_cmc(
                        cmc_cos, cos_dir, filename_prefix=f"cmc_{feature_key}_cosine",
                        title=f"CMC (Cosine) - {feature_key}"
                    )
                    # Per-object ReID stats
                    per_obj_df_cos = compute_per_object_reid_stats(
                        test_stats_cosine["results"], test_stats_cosine["present_object_ids"]
                    )
                    per_obj_df_cos.to_csv(os.path.join(cos_dir, "per_object_reid_stats.csv"), index=False)
                    with open(os.path.join(cos_dir, "per_object_reid_stats.md"), "w") as f:
                        f.write(per_obj_df_cos.to_markdown(index=False))
                    # Save summary stats
                    with open(os.path.join(cos_dir, "success_rates.txt"), "w") as f:
                        f.write(f"Success Rate (Detections): {test_stats_cosine['success_rate_detections']:.2f}%\n")
                        f.write(f"Success Rate (Objects): {test_stats_cosine['success_rate_objects']:.2f}%\n")
                        f.write(f"Mean Average Precision (mAP_standard): {test_stats_cosine['mAP_standard']:.4f}\n")
                        f.write(f"Mean Average Precision (mAP_fair): {test_stats_cosine['mAP_fair']:.4f}\n")
                        f.write(f"Total Detections: {test_stats_cosine['total_detections']}\n")
                        f.write(f"Total Objects: {test_stats_cosine['total_objects']}\n")
                        f.write(f"CMC Rank-1: {r1*100:.2f}% | Rank-5: {r5*100 if not np.isnan(r5) else float('nan'):.2f}% | Rank-10: {r10*100 if not np.isnan(r10) else float('nan'):.2f}%\n")

                    print(f"Cosine ({feature_key}) -> SR(det): {test_stats_cosine['success_rate_detections']:.2f}% | SR(obj): {test_stats_cosine['success_rate_objects']:.2f}% | mAP_std: {test_stats_cosine['mAP_standard']:.4f} | mAP_fair: {test_stats_cosine['mAP_fair']:.4f} | CMC@1: {r1*100:.2f}%")

                    summary_rows.append({
                        "feature_set": feature_key,
                        "metric": "cosine_similarity",
                        "loss_type": "n/a",
                        "neg_strategy": "n/a",
                        "success_rate_detections_%": test_stats_cosine['success_rate_detections'],
                        "success_rate_objects_%": test_stats_cosine['success_rate_objects'],
                        "mAP_standard": test_stats_cosine['mAP_standard'],
                        "mAP_fair": test_stats_cosine['mAP_fair'],
                        "total_detections": test_stats_cosine['total_detections'],
                        "total_objects": test_stats_cosine['total_objects'],
                        "cmc_rank1": r1,
                        "cmc_rank5": r5,
                        "cmc_rank10": r10
                    })

                    # 3) Attention training and test evaluation for CE, Triplet and CE->Triplet(TopKStochastic)
                    for loss_type in attention_loss_types:
                        if loss_type == 'triplet':
                            strategies = triplet_neg_strategies
                        elif loss_type == 'triplet_schedule':
                            strategies = ['n/a']
                        else:
                            strategies = ['n/a']

                        for neg_strategy in strategies:
                            if loss_type == 'triplet':
                                print(f"\nTraining Attention model with Loss: {loss_type} (neg_strategy={neg_strategy}) on Feature Set: {feature_key}")
                            else:
                                print(f"\nTraining Attention model with Loss: {loss_type} on Feature Set: {feature_key}")

                            attention_model = train_attention_model(
                                train_memory, train_stats_for_attention["results"], memory_feature_key=feature_key,
                                valid_memory=valid_memory, valid_augmented_results=valid_stats_for_attention["results"],
                                epochs=100, lr=1e-4, device=extractor.device.type, patience=5, min_delta=1e-4,
                                loss_type=loss_type, margin=triplet_margin,
                                neg_strategy=(neg_strategy if loss_type in ['triplet', 'ce_then_triplet'] else 'random'),
                                hard_negatives_k=40,
                                warmup_epochs=warmup_epochs
                            )

                            print(f"Evaluating YOLO on augmented TEST dataset (attention, loss={loss_type}{'' if loss_type == 'ce' else f', neg={neg_strategy}'})...")
                            test_stats_attention = evaluate_with_yolo(
                                augmented_test_dir, yolo_weights, extractor, test_memory,
                                metric="attention_similarity",
                                confidence_threshold=confidence_threshold,
                                attention_model=attention_model,
                                feature_key=feature_key,
                                geometry_extractor=geometry_extractor
                            )

                            att_dir_suffix = f"{feature_key}_attention_{loss_type}" + (f"_{neg_strategy}" if loss_type in ['triplet', 'ce_then_triplet'] else "")
                            att_dir = os.path.join(config_output_dir, att_dir_suffix)
                            os.makedirs(att_dir, exist_ok=True)
                            results_df_att = pd.DataFrame(test_stats_attention["results"])
                            results_df_att.to_csv(os.path.join(att_dir, "results.csv"), index=False)
                            with open(os.path.join(att_dir, "results.md"), "w") as f:
                                f.write(results_df_att.to_markdown(index=False))
                            # Plots
                            plot_success_and_failure_examples(
                                test_stats_attention["results"], test_memory,
                                augmented_test_dir, original_test_dir, att_dir
                            )
                            plot_topk_examples_for_success_and_failure(
                                test_stats_attention["results"], test_memory,
                                augmented_test_dir, original_test_dir,
                                att_dir, top_k=3, feature_key=feature_key,
                                metric="attention_similarity", attention_model=attention_model
                            )
                            # t-SNE using attention-based distance
                            X_mem2, X_aug2 = plot_tsne_memory_and_augmented(
                                test_memory, test_stats_attention["results"], att_dir, metric='attention_similarity', feature_key=feature_key,
                                attention_model=attention_model, device=extractor.device
                            )
                            plot_tsne_memory_thumbnails_at_coords(
                                X_mem2, test_memory, att_dir, metric='attention_similarity', feature_key=feature_key, keep_aspect_ratio=True
                            )
                            # CMC computation and plot
                            gt_ranks_att = [r.get("gt_rank", None) for r in test_stats_attention["results"]]
                            cmc_att = compute_cmc(gt_ranks_att, num_gallery=len(test_memory), max_rank_plot=50)
                            r1_att, r5_att, r10_att = plot_and_save_cmc(
                                cmc_att, att_dir,
                                filename_prefix=f"cmc_{feature_key}_attention_{loss_type}" + (f"_{neg_strategy}" if loss_type in ['triplet','ce_then_triplet'] else ""),
                                title=f"CMC (Attention-{loss_type}{'/' + neg_strategy if loss_type in ['triplet','ce_then_triplet'] else ''}) - {feature_key}"
                            )
                            # Per-object ReID stats
                            per_obj_df_att = compute_per_object_reid_stats(
                                test_stats_attention["results"], test_stats_attention["present_object_ids"]
                            )
                            per_obj_df_att.to_csv(os.path.join(att_dir, "per_object_reid_stats.csv"), index=False)
                            with open(os.path.join(att_dir, "per_object_reid_stats.md"), "w") as f:
                                f.write(per_obj_df_att.to_markdown(index=False))
                            # Save stats
                            with open(os.path.join(att_dir, "success_rates.txt"), "w") as f:
                                f.write(f"Success Rate (Detections): {test_stats_attention['success_rate_detections']:.2f}%\n")
                                f.write(f"Success Rate (Objects): {test_stats_attention['success_rate_objects']:.2f}%\n")
                                f.write(f"Mean Average Precision (mAP_standard): {test_stats_attention['mAP_standard']:.4f}\n")
                                f.write(f"Mean Average Precision (mAP_fair): {test_stats_attention['mAP_fair']:.4f}\n")
                                f.write(f"Total Detections: {test_stats_attention['total_detections']}\n")
                                f.write(f"Total Objects: {test_stats_attention['total_objects']}\n")
                                f.write(f"CMC Rank-1: {r1_att*100:.2f}% | Rank-5: {r5_att*100 if not np.isnan(r5_att) else float('nan'):.2f}% | Rank-10: {r10_att*100 if not np.isnan(r10_att) else float('nan'):.2f}%\n")

                            print(
                                f"Attention ({feature_key}, loss={loss_type}{'' if loss_type == 'ce' else f', neg={neg_strategy}'}) -> "
                                f"SR(det): {test_stats_attention['success_rate_detections']:.2f}% | SR(obj): {test_stats_attention['success_rate_objects']:.2f}% | "
                                f"mAP_std: {test_stats_attention['mAP_standard']:.4f} | mAP_fair: {test_stats_attention['mAP_fair']:.4f} | CMC@1: {r1_att*100:.2f}%"
                            )

                            summary_rows.append({
                                "feature_set": feature_key,
                                "metric": "attention_similarity",
                                "loss_type": loss_type,
                                "neg_strategy": (neg_strategy if loss_type in ['triplet','ce_then_triplet'] else 'n/a'),
                                "success_rate_detections_%": test_stats_attention['success_rate_detections'],
                                "success_rate_objects_%": test_stats_attention['success_rate_objects'],
                                "mAP_standard": test_stats_attention['mAP_standard'],
                                "mAP_fair": test_stats_attention['mAP_fair'],
                                "total_detections": test_stats_attention['total_detections'],
                                "total_objects": test_stats_attention['total_objects'],
                                "cmc_rank1": r1_att,
                                "cmc_rank5": r5_att,
                                "cmc_rank10": r10_att
                            })

                # Save and print comparison table
                summary_df = pd.DataFrame(summary_rows)
                summary_path_csv = os.path.join(config_output_dir, "summary.csv")
                summary_df.to_csv(summary_path_csv, index=False)
                with open(os.path.join(config_output_dir, "summary.md"), "w") as f:
                    f.write(summary_df.to_markdown(index=False))

                print("\n=== Performance Comparison (feature_set vs metric vs loss_type vs neg_strategy) ===")
                print(summary_df.to_markdown(index=False))

                for _, row in summary_df.iterrows():
                    print(
                        f"[{row['feature_set']} | {row['metric']} | {row['loss_type']} | {row['neg_strategy']}] "
                        f"SR(det): {row['success_rate_detections_%']:.2f}% | SR(obj): {row['success_rate_objects_%']:.2f}% | "
                        f"mAP_std: {row['mAP_standard']:.4f} | mAP_fair: {row['mAP_fair']:.4f} | "
                        f"CMC@1: {(row['cmc_rank1']*100 if pd.notna(row['cmc_rank1']) else float('nan')):.2f}% | "
                        f"CMC@5: {(row['cmc_rank5']*100 if pd.notna(row['cmc_rank5']) else float('nan')):.2f}% | "
                        f"CMC@10: {(row['cmc_rank10']*100 if pd.notna(row['cmc_rank10']) else float('nan')):.2f}%"
                    )
