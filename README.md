# XraySafe-YOLO Minimal Reproducibility Package

This repository is the compact reproducibility package for the manuscript:

**XraySafe-YOLO: Lightweight Contour-Guided Detection of Occluded Prohibited Items in X-ray Security Inspection Images**

Repository used in the manuscript:

```text
https://github.com/shcc320/XraySafe-YOLO
```


## What Is Included

- `xraysafe_yolo/`: CGOA and LSAFF PyTorch module implementations.
- `configs/models/`: YOLO11n-based model YAML files used for ablation and final models.
- `configs/datasets/opixray.yaml.template`: OPIXray dataset YAML template.
- `scripts/prepare_opixray.py`: OPIXray conversion to YOLO format.
- `scripts/run_all.py`: shared training/evaluation runner.
- `scripts/recompute_detection_counts.py`: fixed confidence/IoU false-negative counting.
- `scripts/make_paper_tables.py`: table generation helper.
- `scripts/make_paper_assets_v3.py`: paper figure/table asset generation helper.
- `results/`: compact CSV tables reported in the manuscript.
- `metadata/`: sanitized dataset and run-configuration summaries.

## What Is Not Included

The public OPIXray images/annotations, trained weights, large prediction JSON files, and full training run folders are not included. This keeps the repository compact and avoids redistributing dataset files outside their original license terms.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows:

```bat
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Preparation

Download OPIXray from its official project page. Then convert it to YOLO format:

```bash
python scripts/prepare_opixray.py --raw-root /path/to/OPIXray --out-root experiments/opixray_formal/dataset_yolo --val-ratio 0.1 --seed 42
```

Update `configs/datasets/opixray.yaml.template` or the generated dataset YAML so that `path:` points to the converted dataset directory.

## Main Reproduction Path

The manuscript uses:

- image size: 640;
- batch size: 16;
- seed: 42;
- confidence threshold for FN counting: 0.25;
- IoU threshold for FN counting: 0.50;
- final model config: `configs/models/xraysafe_yolo11n_opt_spatial_resgated.yaml`.

Run a smoke test first:

```bash
bash run_smoke_test.sh
```

or:

```bat
run_smoke_test.bat
```

The Windows formal-run helper used for the final paper experiment is included as:

```text
run_opixray_opt_v3_formal_4070.bat
```

Adjust dataset paths and GPU settings before use.

## Reported Result Tables

The compact tables used by the manuscript are in `results/`:

- `integrated_main_pretty.csv`
- `integrated_main_raw.csv`
- `integrated_subset_raw.csv`
- `class_fixed_counts.csv`
- `screening_v2_raw.csv`

The final model row in `integrated_main_pretty.csv` reports:

- precision: 0.8998;
- recall: 0.8652;
- F1: 0.8822;
- mAP@0.5: 0.8967;
- mAP@0.5:0.95: 0.3940;
- fixed-threshold FN: 174;
- fixed-threshold recall: 0.9022;
- FPS: 306.90.


