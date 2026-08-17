#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最小可行實驗：估計 LLM representation 中的 gender bias direction 穩定性

這個版本優先追求：
1. 在中國大陸可下載 / 可跑
2. 實驗邏輯清楚
3. 能產出可用於套磁的圖

輸出：
- layer_stability.png          各 layer 的 theta 穩定性
- best_layer_heatmap.png       最佳 layer 的 theta cosine similarity heatmap
- occupation_projection.png    中性職業 prompt 在最佳 theta 上的投影
- summary.txt                  文字摘要
"""

import os
import sys
import math
import json
import argparse
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 可重現性設定
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# 資料建構
# -----------------------------
def build_gender_dataset() -> Tuple[List[str], List[int], List[Dict]]:
    """
    建立一個平衡的 male / female prompt dataset。
    label: 0 = male, 1 = female

    設計原則：
    - occupation 與 template 完全共享
    - 僅用名字作為顯式 gender signal
    - 先做最小實驗，因此不做 race/age 等更高維度擴充
    """
    male_names = [
        "James", "John", "Robert", "Michael", "William",
        "David", "Richard", "Joseph"
    ]
    female_names = [
        "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth",
        "Barbara", "Susan", "Jessica"
    ]

    occupations = [
        "doctor", "nurse", "engineer", "teacher", "lawyer",
        "scientist", "manager", "designer", "writer", "accountant"
    ]

    templates = [
        "{name} is a {occupation}. {name} said that",
        "{name} works as a {occupation}. Yesterday, {name} mentioned that",
        "{name}, who is a {occupation}, told a colleague that",
        "At work, {name} the {occupation} explained that",
    ]

    prompts, labels, meta = [], [], []

    for occ in occupations:
        for name in male_names:
            for tpl in templates:
                p = tpl.format(name=name, occupation=occ)
                prompts.append(p)
                labels.append(0)
                meta.append({"name": name, "gender": "male", "occupation": occ, "template": tpl})

        for name in female_names:
            for tpl in templates:
                p = tpl.format(name=name, occupation=occ)
                prompts.append(p)
                labels.append(1)
                meta.append({"name": name, "gender": "female", "occupation": occ, "template": tpl})

    return prompts, labels, meta


def build_neutral_prompts() -> List[str]:
    """
    這些 prompt 不包含明確 gender 名字，用來觀察 occupation 在 theta 上的投影。
    """
    occupations = [
        "doctor", "nurse", "engineer", "teacher", "lawyer",
        "scientist", "manager", "designer", "writer", "accountant",
        "programmer", "chef", "journalist", "pharmacist", "architect"
    ]
    prompts = [f"The {occ} said that" for occ in occupations]
    return prompts


# -----------------------------
# 模型載入
# -----------------------------
def maybe_download_from_modelscope(model_id: str, cache_dir: str) -> str:
    """
    透過 ModelScope 下載模型到本地。
    注意：這要求 modelscope 已安裝，且對應 repo 為 Transformers 格式。
    """
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("[Error] 未安裝 modelscope，請先安裝：")
        print("  python3 -m pip install modelscope")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] ModelScope 下載失敗: {e}")
        print("  請確認網絡連接，或使用本地模型路徑 --local_model_dir")
        sys.exit(1)

    print(f"[Info] 正在從 ModelScope 下載模型：{model_id}")
    model_dir = snapshot_download(model_id, cache_dir=cache_dir)
    print(f"[Info] 模型已下載到：{model_dir}")
    return model_dir


def load_model_and_tokenizer(
    local_model_dir: str,
    device: str,
    model_type: str = "causal"
):
    """
    載入 tokenizer 與 LM。

    Args:
        local_model_dir: 本地模型路徑
        device: cuda 或 cpu
        model_type: "causal" (decoder-only) 或 "encoder" ( BERT-style)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 根據 model_type 選擇載入方式
    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if (device == "cuda") else torch.float32,
            low_cpu_mem_usage=True
        )
    elif model_type == "encoder":
        model = AutoModel.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if (device == "cuda") else torch.float32,
            low_cpu_mem_usage=True
        )
    else:
        raise ValueError(f"不支援的 model_type: {model_type}，請使用 'causal' 或 'encoder'")

    model.to(device)
    model.eval()
    return tokenizer, model


# -----------------------------
# Hidden state 抽取
# -----------------------------
@torch.no_grad()
def extract_last_token_hidden_states(
    prompts: List[str],
    tokenizer,
    model,
    device: str,
    model_type: str = "causal",
    batch_size: int = 8,
    max_length: int = 128
) -> np.ndarray:
    """
    回傳 shape = [N, num_layers, hidden_dim]

    說明：
    - causal LM: hidden_states[0] 是 embedding output，hidden_states[1:] 是 transformer 輸出
    - encoder: hidden_states 包含 all layers
    - 我們取每個樣本「最後一個非 padding token」的 hidden state
    """
    all_features = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting hidden states"):
        batch_prompts = prompts[start:start + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 根據模型類型選擇輸出方式
        if model_type == "causal":
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]  # 跳過 embedding output
        else:  # encoder
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # encoder 使用所有層

        attention_mask = inputs["attention_mask"]

        # 每個樣本最後一個有效 token 的 index
        last_indices = attention_mask.sum(dim=1) - 1

        # 對每一層取最後 token
        layer_features = []
        for hs in hidden_states:
            # hs: [B, T, D]
            batch_idx = torch.arange(hs.size(0), device=hs.device)
            last_h = hs[batch_idx, last_indices, :]  # [B, D]
            layer_features.append(last_h.cpu().float().numpy())

        # 堆疊成 [B, L, D]
        batch_features = np.stack(layer_features, axis=1)
        all_features.append(batch_features)

    return np.concatenate(all_features, axis=0)


# -----------------------------
# Probe 與穩定性分析
# -----------------------------
def normalize_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        return v
    return v / norm


def fit_probe_get_theta(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    訓練 logistic regression probe，回傳：
    - 單位化 theta
    - 訓練集 accuracy（僅供快速檢查，不作正式泛化指標）
    """
    clf = LogisticRegression(
        C=C,
        max_iter=2000,
        solver="liblinear",
        random_state=42
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    acc = accuracy_score(y, pred)
    theta = clf.coef_[0].astype(np.float64)
    theta = normalize_vector(theta)
    return theta, acc


def cross_val_accuracy(X: np.ndarray, y: np.ndarray, n_splits: int = 5, C: float = 1.0) -> float:
    """
    對每層估計簡單泛化能力，避免只看 stability 不看可分性。
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for tr_idx, te_idx in skf.split(X, y):
        clf = LogisticRegression(
            C=C,
            max_iter=2000,
            solver="liblinear",
            random_state=42
        )
        clf.fit(X[tr_idx], y[tr_idx])
        pred = clf.predict(X[te_idx])
        scores.append(accuracy_score(y[te_idx], pred))

    return float(np.mean(scores))


def bootstrap_thetas_for_layer(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 20,
    sample_ratio: float = 0.8,
    C: float = 1.0,
    seed: int = 42
) -> np.ndarray:
    """
    對某一層做 bootstrap，得到多個 theta。
    回傳 shape = [B, hidden_dim]
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    thetas = []

    # 分類平衡抽樣：分別抽 male / female，避免抽樣失衡
    idx0 = np.where(np.array(y) == 0)[0]
    idx1 = np.where(np.array(y) == 1)[0]

    n0 = max(2, int(len(idx0) * sample_ratio))
    n1 = max(2, int(len(idx1) * sample_ratio))

    for b in range(n_bootstrap):
        s0 = rng.choice(idx0, size=n0, replace=True)
        s1 = rng.choice(idx1, size=n1, replace=True)
        idx = np.concatenate([s0, s1])
        rng.shuffle(idx)

        theta, _ = fit_probe_get_theta(X[idx], np.array(y)[idx], C=C)
        thetas.append(theta)

    return np.stack(thetas, axis=0)


def mean_offdiag_cosine(sim_mat: np.ndarray) -> float:
    """
    計算 cosine similarity matrix 的非對角平均值，
    作為「這一層的 theta 穩定性」。
    """
    n = sim_mat.shape[0]
    if n <= 1:
        return 1.0
    mask = ~np.eye(n, dtype=bool)
    return float(sim_mat[mask].mean())


# -----------------------------
# 視覺化
# -----------------------------
def save_layer_stability_plot(stability: List[float], cv_accs: List[float], out_path: str) -> None:
    layers = np.arange(1, len(stability) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(layers, stability, marker="o", label="Theta stability (mean off-diag cosine)")
    plt.plot(layers, cv_accs, marker="s", label="CV probe accuracy")
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title("Layer-wise Stability of Bias Direction")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_heatmap(sim_mat: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(sim_mat, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine similarity")
    plt.title(title)
    plt.xlabel("Bootstrap run")
    plt.ylabel("Bootstrap run")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_projection_barplot(occupations: List[str], scores: List[float], out_path: str) -> None:
    order = np.argsort(scores)
    occ_sorted = [occupations[i] for i in order]
    scores_sorted = [scores[i] for i in order]

    plt.figure(figsize=(10, 6))
    plt.barh(occ_sorted, scores_sorted)
    plt.xlabel("Projection on bias direction")
    plt.title("Neutral Occupation Prompts Projected onto Best-Layer Theta")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# 主流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LLM Bias Direction Stability Analysis MVP"
    )
    parser.add_argument("--local_model_dir", type=str, default="",
                        help="本地模型目錄；若留空且提供 model_id，則嘗試從 ModelScope 下載")
    parser.add_argument("--model_id", type=str, default="",
                        help="ModelScope repo id，例如 Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model_type", type=str, default="causal", choices=["causal", "encoder"],
                        help="模型類型：causal (decoder-only) 或 encoder (BERT-style)")
    parser.add_argument("--cache_dir", type=str, default="./model_cache",
                        help="ModelScope 下載快取目錄")
    parser.add_argument("--output_dir", type=str, default="./outputs_bias_mvp",
                        help="輸出目錄")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--max_length", type=int, default=128,
                        help="最大序列長度")
    parser.add_argument("--n_bootstrap", type=int, default=20,
                        help="Bootstrap 抽樣次數")
    parser.add_argument("--sample_ratio", type=float, default=0.8,
                        help="每次 bootstrap 的抽樣比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="隨機種子")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 決定模型來源
    if args.local_model_dir:
        local_model_dir = args.local_model_dir
    else:
        if not args.model_id:
            print("[Error] 請提供 --local_model_dir 或 --model_id 其中之一")
            sys.exit(1)
        local_model_dir = maybe_download_from_modelscope(args.model_id, args.cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Device: {device}")
    print(f"[Info] Model dir: {local_model_dir}")
    print(f"[Info] Model type: {args.model_type}")

    tokenizer, model = load_model_and_tokenizer(
        local_model_dir,
        device,
        model_type=args.model_type
    )

    # 1) 建立資料
    prompts, labels, meta = build_gender_dataset()
    labels = np.array(labels, dtype=np.int64)
    print(f"[Info] Dataset size: {len(prompts)} samples")

    # 2) 取所有樣本的各層 hidden states
    features = extract_last_token_hidden_states(
        prompts=prompts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_type=args.model_type,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    # features shape: [N, L, D]
    n_samples, n_layers, hidden_dim = features.shape
    print(f"[Info] features shape = {features.shape}")
    print(f"[Info] Hidden dimension = {hidden_dim}, Num layers = {n_layers}")

    # 3) 逐層做 CV probe accuracy + bootstrap theta stability
    layer_stability = []
    layer_cv_acc = []
    layer_theta_means = []
    layer_theta_all = []

    print("[Info] 開始逐層穩定性分析...")
    for layer_idx in tqdm(range(n_layers), desc="Analyzing layers"):
        X = features[:, layer_idx, :]

        # 這個分數代表「這一層能不能分 male/female context」
        cv_acc = cross_val_accuracy(X, labels, n_splits=5, C=1.0)

        # 這組 theta 代表「重抽資料後，方向是否穩定」
        thetas = bootstrap_thetas_for_layer(
            X=X,
            y=labels,
            n_bootstrap=args.n_bootstrap,
            sample_ratio=args.sample_ratio,
            C=1.0,
            seed=args.seed + layer_idx
        )

        sim_mat = cosine_similarity(thetas)
        stability = mean_offdiag_cosine(sim_mat)

        # 平均 theta 作為這層的代表方向
        theta_mean = normalize_vector(thetas.mean(axis=0))

        layer_cv_acc.append(cv_acc)
        layer_stability.append(stability)
        layer_theta_means.append(theta_mean)
        layer_theta_all.append(thetas)

    layer_cv_acc = np.array(layer_cv_acc)
    layer_stability = np.array(layer_stability)

    # 4) 選最佳 layer
    # 這裡用簡單綜合分數：stability * cv_acc
    combined_score = layer_stability * layer_cv_acc
    best_layer = int(np.argmax(combined_score))
    best_thetas = layer_theta_all[best_layer]
    best_sim_mat = cosine_similarity(best_thetas)
    best_theta = layer_theta_means[best_layer]

    print(f"\n[Result] Best layer = {best_layer + 1}")
    print(f"[Result] Best layer stability = {layer_stability[best_layer]:.4f}")
    print(f"[Result] Best layer CV accuracy = {layer_cv_acc[best_layer]:.4f}")
    print(f"[Result] Best layer combined score = {combined_score[best_layer]:.4f}")

    # 5) 中性 occupation prompt 投影
    neutral_prompts = build_neutral_prompts()
    neutral_features = extract_last_token_hidden_states(
        prompts=neutral_prompts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_type=args.model_type,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    neutral_X = neutral_features[:, best_layer, :]
    projection_scores = neutral_X @ best_theta

    occupations = [p.replace("The ", "").replace(" said that", "") for p in neutral_prompts]

    # 6) 存圖
    save_layer_stability_plot(
        stability=layer_stability.tolist(),
        cv_accs=layer_cv_acc.tolist(),
        out_path=os.path.join(args.output_dir, "layer_stability.png")
    )

    save_heatmap(
        sim_mat=best_sim_mat,
        out_path=os.path.join(args.output_dir, "best_layer_heatmap.png"),
        title=f"Best Layer Theta Stability Heatmap (Layer {best_layer + 1})"
    )

    save_projection_barplot(
        occupations=occupations,
        scores=projection_scores.tolist(),
        out_path=os.path.join(args.output_dir, "occupation_projection.png")
    )

    # 7) 存 summary
    summary = {
        "model_dir": local_model_dir,
        "model_type": args.model_type,
        "device": device,
        "n_samples": int(n_samples),
        "n_layers": int(n_layers),
        "hidden_dim": int(hidden_dim),
        "best_layer_1_indexed": int(best_layer + 1),
        "best_layer_stability": float(layer_stability[best_layer]),
        "best_layer_cv_accuracy": float(layer_cv_acc[best_layer]),
        "best_layer_combined_score": float(combined_score[best_layer]),
        "top5_layers_by_combined_score": [
            {
                "layer": int(i + 1),
                "stability": float(layer_stability[i]),
                "cv_accuracy": float(layer_cv_acc[i]),
                "combined_score": float(combined_score[i]),
            }
            for i in np.argsort(combined_score)[::-1][:5]
        ],
        "occupation_projection": [
            {"occupation": occ, "projection": float(score)}
            for occ, score in zip(occupations, projection_scores.tolist())
        ]
    }

    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n[Done] 所有輸出已存到：{args.output_dir}")
    print(f"  - layer_stability.png")
    print(f"  - best_layer_heatmap.png")
    print(f"  - occupation_projection.png")
    print(f"  - summary.txt")


if __name__ == "__main__":
    main()