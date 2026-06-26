from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LATEX_ROOT = REPO_ROOT.parent / "xraysafe_latex"
LATEX_FIGURES = LATEX_ROOT / "figures"
TABLE_OUT = REPO_ROOT / "paper_tables" / "opixray_integrated_v3"
DATASET_ROOT = REPO_ROOT / "experiments" / "opixray_opt_v3_formal" / "dataset_yolo"

CONF_THRES = 0.25
IOU_THRES = 0.50

CLASS_NAMES = [
    "folding_knife",
    "straight_knife",
    "scissor",
    "utility_knife",
    "multi_tool_knife",
]

DISPLAY = {
    "yolo11n": "YOLO11n",
    "xraysafe_yolo11n_cgoa": "+CGOA",
    "xraysafe_yolo11n_lsaff": "+LSAFF",
    "xraysafe_yolo11n_full": "CGOA+LSAFF\n(original)",
    "xraysafe_yolo11n_opt_resgated": "RG-LSAFF\n(all scales)",
    "xraysafe_yolo11n_opt_p34_resgated": "CGOA+RG-LSAFF\n(P3/P4)",
    "xraysafe_yolo11n_opt_spatial_resgated": "XraySafe-YOLO\n(final)",
}

WEIGHTS = {
    "YOLO11n": REPO_ROOT / "experiments" / "opixray_ablation" / "runs_train" / "yolo11n" / "weights" / "best.pt",
    "+CGOA": REPO_ROOT / "experiments" / "opixray_ablation" / "runs_train" / "xraysafe_yolo11n_cgoa" / "weights" / "best.pt",
    "XraySafe-YOLO final": REPO_ROOT / "experiments" / "opixray_opt_v3_formal" / "runs_train" / "xraysafe_yolo11n_opt_spatial_resgated" / "weights" / "best.pt",
}

PAPER_ORDER = [
    "yolo11n",
    "xraysafe_yolo11n_cgoa",
    "xraysafe_yolo11n_lsaff",
    "xraysafe_yolo11n_full",
    "xraysafe_yolo11n_opt_p34_resgated",
    "xraysafe_yolo11n_opt_spatial_resgated",
]


@dataclass
class Box:
    cls: int
    xyxy: Tuple[float, float, float, float]
    conf: float | None = None


def ensure_dirs() -> None:
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.mkdir(parents=True, exist_ok=True)


def style_axes(ax) -> None:
    ax.grid(axis="y", color="#d9dde3", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#a8afb9")
    ax.spines["bottom"].set_color("#a8afb9")
    ax.tick_params(colors="#2e3440", labelsize=8)


def savefig(name: str) -> None:
    path = LATEX_FIGURES / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(path)


def add_box(ax, xy, wh, text, fc="#f7fbff", ec="#4c78a8", fontsize=9, lw=1.2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#1f2933")
    return patch


def add_arrow(ax, start, end, color="#222222", lw=1.4, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    return arrow


def make_architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "input": ("#fff7df", "#c49a28"),
        "backbone": ("#f4fbf1", "#6aa84f"),
        "neck": ("#fff2fb", "#cc79a7"),
        "head": ("#fff8ec", "#e08a2e"),
        "base": ("#eadff5", "#8e63af"),
        "feat": ("#e5f2ff", "#4f81bd"),
        "cgoa": ("#ffd9cc", "#e6550d"),
        "lsaff": ("#dff8fa", "#1aa6b7"),
        "detect": ("#f7d4e8", "#c51b7d"),
    }

    def frame(x, y, w, h, label, fc, ec):
        ax.add_patch(Rectangle((x, y), w, h, linewidth=1.35, edgecolor=ec, facecolor=fc, alpha=0.30))
        ax.text(x + w / 2, y + h + 0.025, label, ha="center", va="center", fontsize=10, weight="bold", color=ec)

    def node(x, y, w, h, label, fc, ec, fontsize=8.2, weight="normal", lw=1.0):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, weight=weight, color="#1f2933")
        return patch

    def arrow(start, end, color="#3a3a3a", lw=1.15, rad=0.0, style="-|>", ls="-"):
        arr = FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=lw,
            color=color,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arr)

    ax.text(0.50, 0.965, "XraySafe-YOLO final architecture", ha="center", va="center", fontsize=13, weight="bold")

    frame(0.135, 0.13, 0.20, 0.74, "Backbone", *colors["backbone"])
    frame(0.375, 0.13, 0.22, 0.74, "Enhanced branches", *colors["neck"])
    frame(0.635, 0.13, 0.15, 0.74, "Fusion", "#e9fbfd", "#1aa6b7")
    frame(0.825, 0.13, 0.15, 0.74, "Head", *colors["head"])

    node(0.025, 0.47, 0.085, 0.08, "Input\n640x640", *colors["input"], fontsize=8.3)
    # small image stack icon
    for dx, dy, fc in [(0.000, 0.000, "#f8d36b"), (0.010, -0.008, "#f2c14e"), (0.020, -0.016, "#e8b23c")]:
        ax.add_patch(Rectangle((0.052 + dx, 0.575 + dy), 0.032, 0.030, facecolor=fc, edgecolor="#8a7a2d", linewidth=0.6))
    arrow((0.110, 0.51), (0.135, 0.51))

    back_x, bw, bh = 0.185, 0.10, 0.055
    backbone_nodes = [
        ("Conv", 0.80, "#e8dff2"),
        ("C3k2", 0.70, "#d7ecff"),
        ("Conv", 0.60, "#e8dff2"),
        ("C3k2", 0.50, "#d7ecff"),
        ("SPPF", 0.40, "#ffe5b4"),
        ("C2PSA", 0.30, "#d7ecff"),
        ("P5\n1/32", 0.20, "#d7ecff"),
    ]
    for idx, (lab, y, fc) in enumerate(backbone_nodes):
        node(back_x, y, bw, bh, lab, fc, "#7c6aa6", fontsize=7.6, weight=("bold" if lab.startswith("P5") else "normal"))
        if idx > 0:
            prev_y = backbone_nodes[idx - 1][1]
            arrow((back_x + bw / 2, prev_y), (back_x + bw / 2, y + bh), lw=1.0)

    row_y = [0.70, 0.50, 0.30]
    p_labs = ["P3\n1/8", "P4\n1/16", "P5\n1/32"]
    source_y = [0.70 + bh / 2, 0.50 + bh / 2, 0.20 + bh / 2]
    for y, lab, sy in zip(row_y, p_labs, source_y):
        node(0.395, y, 0.075, 0.055, lab, *colors["feat"], fontsize=7.6, weight="bold")
        arrow((back_x + bw, sy), (0.395, y + 0.027), rad=0.04 if y < 0.35 else (-0.03 if y > 0.65 else 0.0))

    branch_specs = [
        (0.495, 0.70, "CGOA\ncontour", colors["cgoa"], 7.4),
        (0.495, 0.50, "CGOA\ncontour", colors["cgoa"], 7.4),
        (0.495, 0.30, "P5 bypass\nsemantic feature", ("#f4f4f4", "#7f7f7f"), 7.2),
    ]
    for x, y, lab, color_pair, fs in branch_specs:
        node(x, y, 0.085, 0.055, lab, *color_pair, fontsize=fs, weight="bold" if "CGOA" in lab else "normal")
        arrow((0.470, y + 0.027), (x, y + 0.027))

    node(
        0.645,
        0.255,
        0.13,
        0.50,
        "Residual-gated\nspatial LSAFF\n\nAlign features\n$\\phi_i(F_i)$\n\nSpatial weights\n$\\alpha_{k,i}(x,y)$\n\nGate + identity\n$\\sigma(g_k),\\psi_k(F_k)$",
        *colors["lsaff"],
        fontsize=7.4,
        weight="bold",
        lw=1.3,
    )
    for y in row_y:
        arrow((0.580, y + 0.027), (0.645, y + 0.027))

    for y, out in zip(row_y, ["$P_3^{out}$", "$P_4^{out}$", "$P_5^{out}$"]):
        node(0.835, y, 0.062, 0.055, out, "#f3f8ff", "#4f81bd", fontsize=7.8, weight="bold")
        node(0.910, y, 0.047, 0.055, "Box\nCls", *colors["detect"], fontsize=7.0, weight="bold")
        arrow((0.775, y + 0.027), (0.835, y + 0.027))
        arrow((0.897, y + 0.027), (0.910, y + 0.027))

    for ext in ["png", "pdf"]:
        path = LATEX_FIGURES / f"fig_architecture.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        print(path)
    plt.close(fig)


def make_cgoa_figure() -> None:
    fig, ax = plt.subplots(figsize=(9.8, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.50, 0.94, "Contour-Guided Occlusion Attention (CGOA)", ha="center", va="center", fontsize=13, weight="bold")

    colors = {
        "input": ("#e9fbfd", "#0f8b8d"),
        "contour": ("#e9f7ff", "#1f9acb"),
        "channel": ("#eefbea", "#2ca25f"),
        "fusion": ("#f5efff", "#7b3fc6"),
        "output": ("#fffdf7", "#4a4a4a"),
    }

    def node(x, y, w, h, label, fc, ec, fontsize=9.2, weight="normal", lw=1.25):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.014,rounding_size=0.014",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color="#1f2933", weight=weight)
        return patch

    def arrow(start, end, color="#2f2f2f", lw=1.25, rad=0.0):
        arr = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arr)

    node(0.045, 0.42, 0.135, 0.16, "Input feature\n$F\\in\\mathbb{R}^{C\\times H\\times W}$", *colors["input"], fontsize=9.5, weight="bold")
    split_x, split_y = 0.235, 0.50
    ax.plot([0.180, split_x], [split_y, split_y], color="#2f2f2f", linewidth=1.25)
    ax.scatter([split_x], [split_y], s=34, color="#2f2f2f", zorder=4)

    node(
        0.315,
        0.66,
        0.18,
        0.14,
        "Contour branch\nlocal variation\nboundary response",
        *colors["contour"],
        fontsize=9.2,
        weight="bold",
    )
    node(0.570, 0.66, 0.16, 0.14, "Spatial response\n$M_c\\in\\mathbb{R}^{1\\times H\\times W}$", *colors["contour"], fontsize=9.4)

    node(
        0.315,
        0.20,
        0.18,
        0.14,
        "Channel branch\nGAP + GMP\nshared transform",
        *colors["channel"],
        fontsize=9.2,
        weight="bold",
    )
    node(0.570, 0.20, 0.16, 0.14, "Channel weights\n$A_{ch}\\in\\mathbb{R}^{C\\times1\\times1}$", *colors["channel"], fontsize=9.4)

    node(
        0.775,
        0.37,
        0.20,
        0.25,
        "Feature reweighting\n\n$F'=F\\otimes(1+\\sigma(M_c))\\otimes A_{ch}$\n\nidentity path\npreserves $F$",
        *colors["fusion"],
        fontsize=8.6,
        weight="bold",
        lw=1.35,
    )
    node(0.815, 0.13, 0.13, 0.10, "Enhanced feature\n$F'\\in\\mathbb{R}^{C\\times H\\times W}$", *colors["output"], fontsize=9.2)

    arrow((split_x, split_y), (0.315, 0.73), rad=0.10)
    arrow((split_x, split_y), (0.315, 0.27), rad=-0.10)
    arrow((split_x, split_y), (0.775, 0.50), color="#5f6368", lw=1.1)
    arrow((0.495, 0.73), (0.570, 0.73))
    arrow((0.495, 0.27), (0.570, 0.27))
    arrow((0.730, 0.73), (0.775, 0.58), rad=-0.08)
    arrow((0.730, 0.27), (0.775, 0.42), rad=0.08)
    arrow((0.880, 0.37), (0.880, 0.23))

    ax.text(0.235, 0.545, "split", ha="center", va="bottom", fontsize=8.0, color="#5f6368")
    ax.text(0.50, 0.075, "No extra contour labels are required; the contour response is learned from the input feature map.", ha="center", va="center", fontsize=9.0, color="#4b5563")

    for ext in ["png", "pdf"]:
        path = LATEX_FIGURES / f"fig_cgoa.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        print(path)
    plt.close(fig)


def make_lsaff_figure() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ys = [0.72, 0.50, 0.28]
    labels = ["P3 high resolution", "P4 mid level", "P5 semantic"]
    for y, lab in zip(ys, labels):
        add_box(ax, (0.04, y - 0.045), (0.12, 0.09), lab, fc="#eef9ff", ec="#00a6d6", fontsize=8.5)
        add_box(ax, (0.24, y - 0.045), (0.16, 0.09), f"Align + transform\n$\\phi_i(F_i)$", fc="#f7f0ff", ec="#9966cc", fontsize=8.5)
        add_arrow(ax, (0.16, y), (0.24, y))

    add_box(ax, (0.49, 0.40), (0.16, 0.20), "Spatial scale\nweights\n$\\alpha_{k,i}(x,y)$", fc="#fff6df", ec="#d59f00", fontsize=9)
    for y in ys:
        add_arrow(ax, (0.40, y), (0.49, 0.50), rad=(0.15 if y > 0.5 else (-0.15 if y < 0.5 else 0)))

    add_box(ax, (0.50, 0.14), (0.14, 0.10), "Weighted\ncross-scale sum", fc="#fffdf0", ec="#d59f00", fontsize=8.5)
    add_arrow(ax, (0.57, 0.40), (0.57, 0.24))
    add_box(ax, (0.72, 0.14), (0.11, 0.10), "Residual gate\n$\\sigma(g_k)$", fc="#fff4f2", ec="#d65f5f", fontsize=8.5)
    add_arrow(ax, (0.64, 0.19), (0.72, 0.19))

    add_box(ax, (0.50, 0.76), (0.14, 0.10), "Target-scale\nidentity\n$\\psi_k(F_k)$", fc="#eefbea", ec="#5aae61", fontsize=8.5)
    add_box(ax, (0.75, 0.50), (0.10, 0.10), "Add", fc="#f5f5f5", ec="#666666", fontsize=10)
    add_arrow(ax, (0.64, 0.81), (0.75, 0.56), rad=-0.08)
    add_arrow(ax, (0.83, 0.19), (0.80, 0.50), rad=0.08)

    add_box(ax, (0.90, 0.50), (0.08, 0.10), "$F_k^{out}$\nDetect", fc="#eef6ff", ec="#4878d0", fontsize=8.5)
    add_arrow(ax, (0.85, 0.55), (0.90, 0.55))

    ax.text(
        0.50,
        0.04,
        "$F_k^{out}=\\psi_k(F_k)+\\sigma(g_k)\\sum_i \\alpha_{k,i}(x,y)\\phi_i(F_i),\\quad k\\in\\{3,4,5\\}$",
        ha="center",
        va="center",
        fontsize=12,
        color="#1f2933",
    )
    for ext in ["png", "pdf"]:
        path = LATEX_FIGURES / f"fig_lsaff.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(path)
    plt.close(fig)


def read_main_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(REPO_ROOT / "experiments" / "opixray_ablation" / "metrics" / "main_comparison.csv")
    screen = pd.read_csv(REPO_ROOT / "experiments" / "opixray_opt_v2_screen_v3" / "metrics" / "main_comparison.csv")
    formal = pd.read_csv(REPO_ROOT / "experiments" / "opixray_opt_v3_formal" / "metrics" / "main_comparison.csv")
    return old, screen, formal


def read_subset_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(REPO_ROOT / "experiments" / "opixray_ablation" / "metrics" / "subset_comparison.csv")
    formal = pd.read_csv(REPO_ROOT / "experiments" / "opixray_opt_v3_formal" / "metrics" / "subset_comparison.csv")
    return old, formal


def integrated_main() -> pd.DataFrame:
    old, _, formal = read_main_tables()
    df = pd.concat([old, formal], ignore_index=True)
    df = df[df["model"].isin(PAPER_ORDER)].copy()
    df["order"] = df["model"].map({m: i for i, m in enumerate(PAPER_ORDER)})
    df = df.sort_values("order").drop(columns=["order"])
    return df


def export_integrated_tables() -> None:
    df = integrated_main()
    cols = [
        "model",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
        "false_negative_count",
        "fixed_count_precision",
        "fixed_count_recall",
        "params",
        "gflops",
        "model_size_mb",
        "latency_ms_img",
        "fps",
        "epochs",
    ]
    out = df[cols].copy()
    out.insert(1, "display_name", out["model"].map(DISPLAY).str.replace("\n", " ", regex=False))
    out.to_csv(TABLE_OUT / "integrated_main_raw.csv", index=False)

    pretty = out.copy()
    rename = {
        "display_name": "Model",
        "precision": "P",
        "recall": "R",
        "f1": "F1",
        "map50": "mAP50",
        "map50_95": "mAP50-95",
        "false_negative_count": "FN",
        "fixed_count_precision": "Fixed P",
        "fixed_count_recall": "Fixed R",
        "params": "Params",
        "gflops": "GFLOPs",
        "model_size_mb": "MB",
        "latency_ms_img": "ms/img",
        "fps": "FPS",
        "epochs": "Epochs",
    }
    pretty = pretty.drop(columns=["model"]).rename(columns=rename)
    for c in ["P", "R", "F1", "mAP50", "mAP50-95", "Fixed P", "Fixed R"]:
        pretty[c] = pretty[c].map(lambda x: f"{float(x):.4f}")
    for c in ["GFLOPs", "MB", "ms/img", "FPS"]:
        pretty[c] = pretty[c].map(lambda x: f"{float(x):.2f}")
    pretty["Params"] = pretty["Params"].map(lambda x: f"{float(x) / 1e6:.3f}")
    pretty["FN"] = pretty["FN"].map(lambda x: f"{int(round(float(x)))}")
    pretty["Epochs"] = pretty["Epochs"].map(lambda x: f"{int(round(float(x)))}")
    pretty.to_csv(TABLE_OUT / "integrated_main_pretty.csv", index=False)
    pretty.to_markdown(TABLE_OUT / "integrated_main_pretty.md", index=False)
    (TABLE_OUT / "integrated_main_pretty.tex").write_text(pretty.to_latex(index=False), encoding="utf-8")

    _, screen, _ = read_main_tables()
    screen_cols = [
        "model",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
        "false_negative_count",
        "fixed_count_recall",
        "latency_ms_img",
        "fps",
        "epochs",
    ]
    screen_out = screen[screen_cols].copy()
    screen_out.insert(1, "display_name", screen_out["model"].map(DISPLAY).str.replace("\n", " ", regex=False))
    screen_out.to_csv(TABLE_OUT / "screening_v2_raw.csv", index=False)
    screen_out.to_markdown(TABLE_OUT / "screening_v2_raw.md", index=False)

    old_sub, formal_sub = read_subset_tables()
    sub = pd.concat([old_sub, formal_sub], ignore_index=True)
    sub = sub[sub["model"].isin(["yolo11n", "xraysafe_yolo11n_cgoa", "xraysafe_yolo11n_opt_spatial_resgated"])].copy()
    sub["display_name"] = sub["model"].map(DISPLAY).str.replace("\n", " ", regex=False)
    sub.to_csv(TABLE_OUT / "integrated_subset_raw.csv", index=False)


def plot_main_metrics() -> None:
    df = integrated_main()
    labels = [DISPLAY[m] for m in df["model"]]
    metrics = ["precision", "recall", "f1", "map50", "map50_95", "fixed_count_recall"]
    metric_labels = ["Precision", "Recall", "F1", "mAP50", "mAP50-95", "Fixed Recall"]
    colors = ["#4878d0", "#6acc64", "#d65f5f", "#b47cc7", "#c4ad66", "#4c9a9a"]
    x = np.arange(len(labels))
    width = 0.12
    plt.figure(figsize=(11.6, 4.6))
    ax = plt.gca()
    for i, (m, lab) in enumerate(zip(metrics, metric_labels)):
        ax.bar(x + (i - 2.5) * width, df[m].astype(float), width, label=lab, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0.35, 0.94)
    ax.set_ylabel("Metric value")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    style_axes(ax)
    savefig("fig_overall_metrics_integrated.png")


def plot_fn_recall() -> None:
    df = integrated_main()
    labels = [DISPLAY[m] for m in df["model"]]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(10.6, 4.2))
    bars = ax1.bar(x, df["false_negative_count"].astype(float), color="#d65f5f", width=0.58, label="Fixed FN")
    ax1.set_ylabel("False negatives (lower is better)", color="#8a2f2f")
    ax1.tick_params(axis="y", labelcolor="#8a2f2f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    style_axes(ax1)
    for b in bars:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, f"{int(b.get_height())}", ha="center", va="bottom", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, df["fixed_count_recall"].astype(float), color="#287c71", marker="o", linewidth=2.2, label="Fixed recall")
    ax2.set_ylim(0.84, 0.91)
    ax2.set_ylabel("Fixed recall", color="#287c71")
    ax2.tick_params(axis="y", labelcolor="#287c71")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#287c71")
    lines, labs = [], []
    for ax in [ax1, ax2]:
        a, l = ax.get_legend_handles_labels()
        lines.extend(a)
        labs.extend(l)
    ax1.legend(lines, labs, frameon=False, loc="upper right")
    savefig("fig_fn_fixed_recall.png")


def plot_efficiency_tradeoff() -> None:
    df = integrated_main()
    plt.figure(figsize=(7.2, 4.8))
    ax = plt.gca()
    for _, row in df.iterrows():
        size = 110 + (float(row["params"]) / 1e6 - 2.4) * 120
        ax.scatter(float(row["fps"]), float(row["map50"]), s=size, alpha=0.82)
        ax.annotate(DISPLAY[row["model"]].replace("\n", " "), (float(row["fps"]), float(row["map50"])),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("FPS")
    ax.set_ylabel("mAP50")
    ax.set_ylim(0.855, 0.902)
    ax.set_xlim(120, max(460, float(df["fps"].max()) + 25))
    style_axes(ax)
    savefig("fig_efficiency_tradeoff.png")


def plot_occlusion() -> None:
    old_sub, formal_sub = read_subset_tables()
    sub = pd.concat([old_sub, formal_sub], ignore_index=True)
    keep = ["yolo11n", "xraysafe_yolo11n_cgoa", "xraysafe_yolo11n_opt_spatial_resgated"]
    sub = sub[sub["model"].isin(keep)].copy()
    subsets = ["ol1", "ol2", "ol3"]
    colors = {"yolo11n": "#4878d0", "xraysafe_yolo11n_cgoa": "#6acc64", "xraysafe_yolo11n_opt_spatial_resgated": "#d65f5f"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric, ylabel, ylim in [
        (axes[0], "recall", "Recall", (0.76, 0.90)),
        (axes[1], "map50", "mAP50", (0.84, 0.915)),
        (axes[2], "false_negative_count", "FN (lower is better)", None),
    ]:
        x = np.arange(len(subsets))
        width = 0.24
        for i, model in enumerate(keep):
            vals = []
            for s in subsets:
                vals.append(float(sub[(sub["model"] == model) & (sub["subset"].str.lower() == s)][metric].iloc[0]))
            ax.bar(x + (i - 1) * width, vals, width=width, label=DISPLAY[model].replace("\n", " "), color=colors[model])
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in subsets])
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=7, ncol=1, loc="lower left")
    savefig("fig_occlusion_level_analysis.png")


def plot_training_curves() -> None:
    runs = {
        "+CGOA": REPO_ROOT / "experiments" / "opixray_ablation" / "runs_train" / "xraysafe_yolo11n_cgoa" / "results.csv",
        "XraySafe-YOLO final": REPO_ROOT / "experiments" / "opixray_opt_v3_formal" / "runs_train" / "xraysafe_yolo11n_opt_spatial_resgated" / "results.csv",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for label, path in runs.items():
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        axes[0].plot(df["epoch"], df["metrics/mAP50(B)"], label=label, linewidth=1.8)
        axes[1].plot(df["epoch"], df["metrics/recall(B)"], label=label, linewidth=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation mAP50")
    axes[0].set_ylim(0.32, 0.93)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation recall")
    axes[1].set_ylim(0.38, 0.90)
    for ax in axes:
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    savefig("fig_training_curves_cgoa_final.png")


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
    except ValueError:
        return image_path.with_suffix(".txt")
    return Path(*parts).with_suffix(".txt")


def yolo_to_xyxy(row: Sequence[float], w: int, h: int) -> Tuple[float, float, float, float]:
    _, xc, yc, bw, bh = row[:5]
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return x1, y1, x2, y2


def read_gt(image_path: Path) -> List[Box]:
    im = Image.open(image_path)
    w, h = im.size
    label_path = label_path_for_image(image_path)
    boxes: List[Box] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        toks = line.split()
        if len(toks) < 5:
            continue
        vals = [float(t) for t in toks[:5]]
        boxes.append(Box(int(vals[0]), yolo_to_xyxy(vals, w, h)))
    return boxes


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_predictions(json_path: Path, conf: float = CONF_THRES) -> Dict[str, List[Box]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    out: Dict[str, List[Box]] = {}
    for row in payload:
        score = float(row.get("score", 0.0))
        if score < conf:
            continue
        file_name = str(row["file_name"])
        x, y, w, h = [float(v) for v in row["bbox"]]
        cls = int(row["category_id"]) - 1
        out.setdefault(file_name, []).append(Box(cls, (x, y, x + w, y + h), score))
    for boxes in out.values():
        boxes.sort(key=lambda b: b.conf or 0.0, reverse=True)
    return out


def load_fixed_predictions(model_name: str) -> Dict[str, List[Box]]:
    """Load or create prediction cache from best.pt with the fixed count protocol."""
    cache = TABLE_OUT / f"fixed_predictions_{model_name.replace('+', 'plus').replace(' ', '_')}.json"
    if not cache.exists():
        from run_all import ensure_ultralytics_patched

        ensure_ultralytics_patched()
        from ultralytics import YOLO

        images = all_test_images()
        model = YOLO(str(WEIGHTS[model_name]))
        rows = []
        chunk_size = 32
        for start in range(0, len(images), chunk_size):
            chunk = images[start:start + chunk_size]
            results = model.predict(
                source=[str(x) for x in chunk],
                imgsz=640,
                conf=CONF_THRES,
                iou=IOU_THRES,
                device="0",
                batch=4,
                save=False,
                verbose=False,
                stream=True,
            )
            for offset, res in enumerate(results):
                image_path = chunk[offset]
                if getattr(res, "boxes", None) is None or res.boxes is None:
                    continue
                xyxy = res.boxes.xyxy.detach().cpu().tolist()
                cls = res.boxes.cls.detach().cpu().tolist()
                confs = res.boxes.conf.detach().cpu().tolist()
                for c, box, score in zip(cls, xyxy, confs):
                    rows.append(
                        {
                            "file_name": image_path.name,
                            "class_id": int(c),
                            "xyxy": [float(v) for v in box],
                            "score": float(score),
                        }
                    )
        cache.write_text(json.dumps(rows), encoding="utf-8")
    payload = json.loads(cache.read_text(encoding="utf-8"))
    out: Dict[str, List[Box]] = {}
    for row in payload:
        out.setdefault(row["file_name"], []).append(
            Box(int(row["class_id"]), tuple(float(v) for v in row["xyxy"]), float(row["score"]))
        )
    for boxes in out.values():
        boxes.sort(key=lambda b: b.conf or 0.0, reverse=True)
    return out


def match_boxes(gt: Sequence[Box], pred: Sequence[Box]) -> tuple[set[int], set[int], List[Tuple[int, int, float]]]:
    candidates: List[Tuple[float, int, int]] = []
    for pi, pb in enumerate(pred):
        for gi, gb in enumerate(gt):
            if pb.cls != gb.cls:
                continue
            ov = iou(pb.xyxy, gb.xyxy)
            if ov >= IOU_THRES:
                candidates.append((ov, pi, gi))
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    pairs: List[Tuple[int, int, float]] = []
    for ov, pi, gi in sorted(candidates, reverse=True):
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)
            pairs.append((pi, gi, ov))
    return matched_pred, matched_gt, pairs


def all_test_images() -> List[Path]:
    return sorted((DATASET_ROOT / "images" / "test").glob("*.jpg"))


def subset_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for subset in ["OL1", "OL2", "OL3"]:
        for p in (DATASET_ROOT / "images" / subset).glob("*.jpg"):
            mapping[p.name] = subset
    return mapping


def class_count_table() -> pd.DataFrame:
    preds = {k: load_fixed_predictions(k) for k in WEIGHTS}
    rows = []
    for model_name, pred_map in preds.items():
        tp = np.zeros(len(CLASS_NAMES), dtype=int)
        fp = np.zeros(len(CLASS_NAMES), dtype=int)
        fn = np.zeros(len(CLASS_NAMES), dtype=int)
        gt_total = np.zeros(len(CLASS_NAMES), dtype=int)
        for image_path in all_test_images():
            gt = read_gt(image_path)
            pred = pred_map.get(image_path.name, [])
            matched_pred, matched_gt, _ = match_boxes(gt, pred)
            for gb in gt:
                gt_total[gb.cls] += 1
            for gi, gb in enumerate(gt):
                if gi in matched_gt:
                    tp[gb.cls] += 1
                else:
                    fn[gb.cls] += 1
            for pi, pb in enumerate(pred):
                if pi not in matched_pred:
                    fp[pb.cls] += 1
        for i, cname in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "model": model_name,
                    "class": cname,
                    "tp": int(tp[i]),
                    "fp": int(fp[i]),
                    "fn": int(fn[i]),
                    "gt": int(gt_total[i]),
                    "fixed_recall": float(tp[i] / gt_total[i]) if gt_total[i] else math.nan,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_OUT / "class_fixed_counts.csv", index=False)
    return df


def plot_class_recall() -> None:
    df = class_count_table()
    models = ["YOLO11n", "+CGOA", "XraySafe-YOLO final"]
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.0))
    for i, model in enumerate(models):
        vals = [float(df[(df["model"] == model) & (df["class"] == cname)]["fixed_recall"].iloc[0]) for cname in CLASS_NAMES]
        fns = [int(df[(df["model"] == model) & (df["class"] == cname)]["fn"].iloc[0]) for cname in CLASS_NAMES]
        axes[0].bar(x + (i - 1) * width, vals, width, label=model)
        axes[1].bar(x + (i - 1) * width, fns, width, label=model)
    for ax, ylabel in [(axes[0], "Fixed recall"), (axes[1], "False negatives")]:
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_NAMES], fontsize=7)
        ax.set_ylabel(ylabel)
        style_axes(ax)
    axes[0].set_ylim(0.65, 1.01)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    savefig("fig_class_fixed_recall_fn.png")


def plot_annotation_stats() -> None:
    rows = []
    for split in ["train", "val", "test", "OL1", "OL2", "OL3"]:
        image_dir = DATASET_ROOT / "images" / split
        for image_path in sorted(image_dir.glob("*.jpg")):
            im = Image.open(image_path)
            w, h = im.size
            for box in read_gt(image_path):
                x1, y1, x2, y2 = box.xyxy
                rows.append(
                    {
                        "split": split,
                        "class": CLASS_NAMES[box.cls],
                        "area": max(0.0, (x2 - x1) * (y2 - y1)) / (w * h),
                        "aspect": (x2 - x1) / max(1e-6, (y2 - y1)),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_OUT / "annotation_statistics.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    cls_counts = df[df["split"].isin(["train", "val", "test"])].groupby("class").size().reindex(CLASS_NAMES)
    axes[0].bar(np.arange(len(CLASS_NAMES)), cls_counts.values, color="#4878d0")
    axes[0].set_xticks(np.arange(len(CLASS_NAMES)))
    axes[0].set_xticklabels([c.replace("_", "\n") for c in CLASS_NAMES], fontsize=7)
    axes[0].set_ylabel("Box count")
    axes[0].set_title("Class annotations")

    test_df = df[df["split"] == "test"]
    axes[1].hist(test_df["area"], bins=30, color="#6acc64", edgecolor="white")
    axes[1].set_xlabel("Normalized box area")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Test-set object size")

    subset_counts = df[df["split"].isin(["OL1", "OL2", "OL3"])].groupby("split").size().reindex(["OL1", "OL2", "OL3"])
    axes[2].bar(np.arange(3), subset_counts.values, color="#d65f5f")
    axes[2].set_xticks(np.arange(3))
    axes[2].set_xticklabels(["OL1", "OL2", "OL3"])
    axes[2].set_ylabel("Box count")
    axes[2].set_title("Occlusion subsets")
    for ax in axes:
        style_axes(ax)
    savefig("fig_annotation_statistics.png")


def get_font(size: int) -> ImageFont.ImageFont:
    for p in [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]:
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def draw_rect(draw: ImageDraw.ImageDraw, box: Sequence[float], color: str, width: int = 4) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    for k in range(width):
        draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)


def draw_boxes_on_image(image_path: Path, gt: Sequence[Box], pred: Sequence[Box] | None, mode: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = get_font(20)
    small_font = get_font(16)
    if mode == "gt":
        for b in gt:
            draw_rect(draw, b.xyxy, "#22a884", width=4)
            label = CLASS_NAMES[b.cls]
            draw.text((b.xyxy[0] + 4, max(2, b.xyxy[1] - 24)), label, fill="#22a884", font=small_font)
    elif mode == "pred":
        pred = list(pred or [])
        matched_pred, matched_gt, _ = match_boxes(gt, pred)
        for gi, b in enumerate(gt):
            if gi not in matched_gt:
                draw_rect(draw, b.xyxy, "#d65f5f", width=4)
                draw.text((b.xyxy[0] + 4, max(2, b.xyxy[1] - 24)), "missed GT", fill="#d65f5f", font=small_font)
        for pi, b in enumerate(pred):
            color = "#00a6d6" if pi in matched_pred else "#e69f00"
            draw_rect(draw, b.xyxy, color, width=4)
            label = f"{CLASS_NAMES[b.cls]} {b.conf:.2f}" if b.conf is not None else CLASS_NAMES[b.cls]
            draw.text((b.xyxy[0] + 4, min(image.height - 22, b.xyxy[1] + 4)), label, fill=color, font=small_font)
    elif mode == "raw":
        draw.text((10, 10), image_path.name, fill="#1f2933", font=font)
    return image


def resize_pad(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    im = image.copy()
    im.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (245, 247, 250))
    canvas.paste(im, ((target_w - im.width) // 2, (target_h - im.height) // 2))
    return canvas


def select_visual_cases() -> tuple[List[Path], List[Path]]:
    cgoa_pred = load_fixed_predictions("+CGOA")
    final_pred = load_fixed_predictions("XraySafe-YOLO final")
    recovered: List[Tuple[int, Path]] = []
    remaining: List[Tuple[int, Path]] = []
    for image_path in all_test_images():
        gt = read_gt(image_path)
        if not gt:
            continue
        _, c_mg, _ = match_boxes(gt, cgoa_pred.get(image_path.name, []))
        _, f_mg, _ = match_boxes(gt, final_pred.get(image_path.name, []))
        c_fn = len(gt) - len(c_mg)
        f_fn = len(gt) - len(f_mg)
        if c_fn > f_fn:
            recovered.append((c_fn - f_fn, image_path))
        if f_fn > 0:
            remaining.append((f_fn, image_path))
    recovered = sorted(recovered, key=lambda x: (-x[0], x[1].name))
    remaining = sorted(remaining, key=lambda x: (-x[0], x[1].name))
    recovered_paths = [p for _, p in recovered[:4]]
    remaining_paths = [p for _, p in remaining[:2]]
    return recovered_paths, remaining_paths


def make_raw_gt_pred_grid() -> None:
    final_pred = load_fixed_predictions("XraySafe-YOLO final")
    recovered, remaining = select_visual_cases()
    cases = (recovered[:3] + remaining[:1])[:4]
    subset_by_name = subset_map()
    cell_w, cell_h = 360, 265
    label_h = 56
    pad = 12
    cols = ["Original", "Ground truth", "Final prediction"]
    canvas = Image.new("RGB", (3 * cell_w + 4 * pad, len(cases) * (cell_h + label_h) + pad), "white")
    draw = ImageDraw.Draw(canvas)
    font = get_font(20)
    small_font = get_font(15)
    for r, image_path in enumerate(cases):
        gt = read_gt(image_path)
        pred = final_pred.get(image_path.name, [])
        _, matched_gt, _ = match_boxes(gt, pred)
        status = "TP" if len(matched_gt) == len(gt) else "remaining FN"
        header = f"{subset_by_name.get(image_path.name, '')} {image_path.name} | {status}"
        y0 = pad + r * (cell_h + label_h)
        draw.text((pad, y0 + 5), header, fill="#1f2933", font=small_font)
        panels = [
            draw_boxes_on_image(image_path, gt, None, "raw"),
            draw_boxes_on_image(image_path, gt, None, "gt"),
            draw_boxes_on_image(image_path, gt, pred, "pred"),
        ]
        for c, panel in enumerate(panels):
            x0 = pad + c * (cell_w + pad)
            if r == 0:
                draw.text((x0 + 4, y0 + 32), cols[c], fill="#1f2933", font=small_font)
            canvas.paste(resize_pad(panel, (cell_w, cell_h)), (x0, y0 + label_h))
    out = LATEX_FIGURES / "fig_raw_gt_prediction_cases.png"
    canvas.save(out, quality=95)
    print(out)


def make_cgoa_final_comparison_grid() -> None:
    cgoa_pred = load_fixed_predictions("+CGOA")
    final_pred = load_fixed_predictions("XraySafe-YOLO final")
    recovered, remaining = select_visual_cases()
    cases = (recovered[:3] + remaining[:1])[:4]
    subset_by_name = subset_map()
    cell_w, cell_h = 360, 265
    label_h = 56
    pad = 12
    cols = ["Ground truth", "+CGOA prediction", "Final prediction"]
    canvas = Image.new("RGB", (3 * cell_w + 4 * pad, len(cases) * (cell_h + label_h) + pad), "white")
    draw = ImageDraw.Draw(canvas)
    small_font = get_font(15)
    for r, image_path in enumerate(cases):
        gt = read_gt(image_path)
        c_pred = cgoa_pred.get(image_path.name, [])
        f_pred = final_pred.get(image_path.name, [])
        _, c_mg, _ = match_boxes(gt, c_pred)
        _, f_mg, _ = match_boxes(gt, f_pred)
        header = (
            f"{subset_by_name.get(image_path.name, '')} {image_path.name} | "
            f"FN: +CGOA {len(gt) - len(c_mg)}, final {len(gt) - len(f_mg)}"
        )
        y0 = pad + r * (cell_h + label_h)
        draw.text((pad, y0 + 5), header, fill="#1f2933", font=small_font)
        panels = [
            draw_boxes_on_image(image_path, gt, None, "gt"),
            draw_boxes_on_image(image_path, gt, c_pred, "pred"),
            draw_boxes_on_image(image_path, gt, f_pred, "pred"),
        ]
        for c, panel in enumerate(panels):
            x0 = pad + c * (cell_w + pad)
            if r == 0:
                draw.text((x0 + 4, y0 + 32), cols[c], fill="#1f2933", font=small_font)
            canvas.paste(resize_pad(panel, (cell_w, cell_h)), (x0, y0 + label_h))
    out = LATEX_FIGURES / "fig_cgoa_final_case_comparison.png"
    canvas.save(out, quality=95)
    print(out)


def copy_standard_ultralytics_figures() -> None:
    src_dir = REPO_ROOT / "experiments" / "opixray_opt_v3_formal" / "runs_val" / "xraysafe_yolo11n_opt_spatial_resgated"
    mapping = {
        "BoxPR_curve.png": "fig_final_pr_curve.png",
        "confusion_matrix_normalized.png": "fig_final_confusion_matrix_normalized.png",
    }
    for src_name, dst_name in mapping.items():
        src = src_dir / src_name
        if src.exists():
            dst = LATEX_FIGURES / dst_name
            shutil.copy2(src, dst)
            print(dst)


def main() -> None:
    ensure_dirs()
    make_architecture_figure()
    make_cgoa_figure()
    make_lsaff_figure()
    export_integrated_tables()
    plot_main_metrics()
    plot_fn_recall()
    plot_efficiency_tradeoff()
    plot_occlusion()
    plot_training_curves()
    plot_annotation_stats()
    plot_class_recall()
    make_raw_gt_pred_grid()
    make_cgoa_final_comparison_grid()
    copy_standard_ultralytics_figures()


if __name__ == "__main__":
    main()
