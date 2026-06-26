from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the repository root importable when this script is executed as
# `python scripts/run_all.py`. This is required for xraysafe_yolo modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import yaml

from common import ensure_dir, log, save_json, read_yaml
from patch_ultralytics import ensure_ultralytics_patched


def import_prepare(dataset: str):
    if dataset.lower() == 'opixray':
        mod = importlib.import_module('prepare_opixray')
        return mod.prepare
    if dataset.lower() == 'pidray':
        mod = importlib.import_module('prepare_pidray')
        return mod.prepare
    raise ValueError(f'Unsupported dataset: {dataset}')


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def clear_cuda_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def metric_from_results(results) -> Dict[str, Any]:
    box = getattr(results, 'box', None)
    out = {}
    # Ultralytics objects change across versions, so access defensively.
    for key, candidates in {
        'precision': ['mp', 'p'],
        'recall': ['mr', 'r'],
        'map50': ['map50'],
        'map50_95': ['map'],
        'map75': ['map75'],
    }.items():
        val = None
        if box is not None:
            for c in candidates:
                if hasattr(box, c):
                    val = getattr(box, c)
                    break
        if hasattr(val, 'item'):
            val = val.item()
        elif isinstance(val, (list, tuple)) and val:
            try:
                val = sum(float(v) for v in val) / len(val)
            except Exception:
                val = None
        out[key] = safe_float(val)
    p, r = out.get('precision'), out.get('recall')
    out['f1'] = (2 * p * r / (p + r)) if p is not None and r is not None and (p + r) > 0 else None
    return out


def count_params(model) -> Optional[int]:
    try:
        return int(sum(p.numel() for p in model.model.parameters()))
    except Exception:
        return None


def dataset_nc(data_yaml: Path) -> Optional[int]:
    """Return class count from a YOLO data YAML."""
    try:
        data = read_yaml(data_yaml)
        if data.get('nc') is not None:
            return int(data['nc'])
        names = data.get('names')
        if isinstance(names, dict):
            return len(names)
        if isinstance(names, list):
            return len(names)
    except Exception:
        return None
    return None


def runtime_model_yaml(model_yaml: Path, data_yaml: Path, project_dir: Path) -> Path:
    """Create an experiment-local YAML with nc aligned to the active dataset.

    Ultralytics otherwise rebuilds YAML models during training when data nc
    differs from model nc.  Aligning nc before loading lets us transfer
    same-shaped Detect head weights from previous OPIXray checkpoints.
    """
    nc = dataset_nc(data_yaml)
    if nc is None:
        return model_yaml
    cfg = read_yaml(model_yaml)
    if int(cfg.get('nc', nc)) == nc:
        return model_yaml
    cfg['nc'] = nc
    out_dir = ensure_dir(project_dir / 'runtime_model_yamls')
    out = out_dir / f'{model_yaml.stem}_nc{nc}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def last_module_by_class_name(module, class_name: str):
    """Find the last submodule whose class name matches class_name."""
    found = None
    for m in module.modules():
        if m.__class__.__name__ == class_name:
            found = m
    return found


def copy_detect_head_by_shape(target_yolo, source_weights: str | Path) -> int:
    """Copy a Detect head from a checkpoint even when layer indices differ.

    The default Ultralytics transfer is key-based, so inserting LSAFF before
    Detect changes indices and prevents Detect weights from being reused.
    This helper copies the local state dict of the last Detect module by key
    and exact shape, which is conservative and transparent.
    """
    try:
        import torch
    except Exception:
        return 0
    source_weights = Path(source_weights)
    if not source_weights.exists() or source_weights.suffix.lower() != '.pt':
        return 0
    try:
        ckpt = torch.load(source_weights, map_location='cpu', weights_only=False)
        source_model = ckpt.get('ema') or ckpt.get('model')
        if source_model is None:
            return 0
        source_model = source_model.float()
        source_detect = last_module_by_class_name(source_model, 'Detect')
        target_detect = last_module_by_class_name(target_yolo.model, 'Detect')
        if source_detect is None or target_detect is None:
            return 0
        source_state = source_detect.state_dict()
        target_state = target_detect.state_dict()
        copied = 0
        with torch.no_grad():
            for key, value in source_state.items():
                if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
                    target_state[key].copy_(value.to(device=target_state[key].device, dtype=target_state[key].dtype))
                    copied += 1
        target_detect.load_state_dict(target_state, strict=False)
        return copied
    except Exception as e:
        log(f'WARNING: Detect head shape-based transfer failed for {source_weights}: {e}')
        return 0


def save_initialized_checkpoint(model, out_path: Path) -> Path:
    """Persist an initialized YOLO model so training starts from exact weights."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    try:
        model.save(str(out_path))
        return out_path
    except Exception as e:
        log(f'WARNING: model.save failed for {out_path}: {e}. Falling back to torch.save.')
    try:
        import copy
        import torch

        m = copy.deepcopy(model.model).half()
        ckpt = {
            'model': m,
            'ema': None,
            'updates': None,
            'optimizer': None,
            'train_args': {},
        }
        torch.save(ckpt, out_path)
        return out_path
    except Exception as e:
        raise RuntimeError(f'Could not save initialized checkpoint {out_path}: {e}') from e




def resolve_split_images(data_yaml: Path, split: str = 'test') -> List[Path]:
    """Resolve image paths from a YOLO data YAML split entry."""
    data = read_yaml(data_yaml)
    root = Path(data.get('path', data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    entry = data.get(split) or data.get('val') or data.get('train')
    if entry is None:
        return []
    entries = entry if isinstance(entry, list) else [entry]
    images: List[Path] = []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for e in entries:
        p = Path(e)
        if not p.is_absolute():
            p = root / p
        if p.is_file() and p.suffix.lower() == '.txt':
            images.extend([Path(line.strip()) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()])
        elif p.is_dir():
            images.extend([x for x in p.rglob('*') if x.suffix.lower() in exts])
        elif p.is_file() and p.suffix.lower() in exts:
            images.append(p)
    return sorted(dict.fromkeys([x.resolve() for x in images if x.exists()]))


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    # Replace the last occurrence of "images" with "labels".
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == 'images':
            parts[i] = 'labels'
            return Path(*parts).with_suffix('.txt')
    return image_path.with_suffix('.txt')


def yolo_label_to_xyxy(row: List[float], w: int, h: int):
    cls, cx, cy, bw, bh = row
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h
    return int(cls), [x1, y1, x2, y2]


def read_gt_boxes(image_path: Path, w: int, h: int):
    lp = label_path_for_image(image_path)
    boxes = []
    if not lp.exists():
        return boxes
    for line in lp.read_text(encoding='utf-8').splitlines():
        toks = line.strip().split()
        if len(toks) < 5:
            continue
        try:
            row = [float(x) for x in toks[:5]]
            boxes.append(yolo_label_to_xyxy(row, w, h))
        except Exception:
            continue
    return boxes


def iou_xyxy(a, b) -> float:
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


def count_detection_errors(model, data_yaml: Path, args) -> Dict[str, Any]:
    """Class-aware greedy TP/FP/FN count at a fixed conf/IoU threshold.

    This complements Ultralytics mAP with a manuscript-friendly FN count. It is
    intentionally simple and transparent: for each image, predictions are greedily
    matched to same-class ground-truth boxes by IoU.
    """
    images = resolve_split_images(data_yaml, split='test')
    if args.max_count_images and args.max_count_images > 0:
        images = images[: args.max_count_images]
    if not images:
        return {'count_error_note': 'No test images resolved for FN/FP/TP counting.'}
    tp = fp = fn = gt_total = pred_total = 0
    try:
        clear_cuda_cache()
        count_batch = max(1, min(int(getattr(args, 'batch', 1) or 1), 4))
        chunk_size = max(count_batch, int(getattr(args, 'count_chunk_size', 32) or 32))
        for start in range(0, len(images), chunk_size):
            chunk = images[start:start + chunk_size]
            results = model.predict(
                source=[str(x) for x in chunk],
                imgsz=args.imgsz,
                conf=args.count_conf,
                iou=args.count_iou,
                device=args.device,
                batch=count_batch,
                save=False,
                verbose=False,
                stream=True,
            )
            for offset, res in enumerate(results):
                idx = start + offset
                img_path = Path(res.path).resolve()
                # Ultralytics may report synthetic names such as image0.jpg when
                # predicting from an in-memory list. Preserve GT lookup by falling
                # back to the original source path in stream order.
                if not label_path_for_image(img_path).exists() and idx < len(images):
                    img_path = images[idx]
                h, w = res.orig_shape[:2]
                gt = read_gt_boxes(img_path, w, h)
                gt_total += len(gt)
                pred = []
                if getattr(res, 'boxes', None) is not None and res.boxes is not None:
                    xyxy = res.boxes.xyxy.detach().cpu().tolist()
                    cls = res.boxes.cls.detach().cpu().tolist()
                    for c, box in zip(cls, xyxy):
                        pred.append((int(c), box))
                pred_total += len(pred)
                matched_gt = set()
                matched_pred = set()
                candidates = []
                for pi, (pc, pb) in enumerate(pred):
                    for gi, (gc, gb) in enumerate(gt):
                        if pc == gc:
                            ov = iou_xyxy(pb, gb)
                            if ov >= args.count_iou:
                                candidates.append((ov, pi, gi))
                for ov, pi, gi in sorted(candidates, reverse=True):
                    if pi not in matched_pred and gi not in matched_gt:
                        matched_pred.add(pi)
                        matched_gt.add(gi)
                tp += len(matched_gt)
                fp += len(pred) - len(matched_pred)
                fn += len(gt) - len(matched_gt)
            del results
            clear_cuda_cache()
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        return {
            'true_positive_count': tp,
            'false_positive_count': fp,
            'false_negative_count': fn,
            'gt_box_count': gt_total,
            'pred_box_count': pred_total,
            'fixed_count_precision': precision,
            'fixed_count_recall': recall,
            'count_conf': args.count_conf,
            'count_iou': args.count_iou,
            'counted_images': len(images),
        }
    except Exception as e:
        return {'count_error_note': f'Fixed-threshold TP/FP/FN counting failed: {e}'}
    finally:
        clear_cuda_cache()


def estimate_gflops(model, imgsz: int) -> Optional[float]:
    """Best-effort GFLOPs extraction across Ultralytics versions."""
    try:
        from ultralytics.utils.torch_utils import get_flops
        val = safe_float(get_flops(model.model, imgsz))
        if val and val > 0:
            return val
    except Exception:
        pass
    try:
        info = model.model.info(verbose=False, imgsz=imgsz)
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            return safe_float(info[-1])
    except Exception:
        pass
    try:
        info = model.info(verbose=False, imgsz=imgsz)
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            return safe_float(info[-1])
    except Exception:
        pass
    return None

def run_train_and_val(model_name: str, data_yaml: Path, project_dir: Path, args) -> Dict[str, Any]:
    # Patch Ultralytics before importing YOLO so YAML files can resolve CGOA/LSAFF.
    ensure_ultralytics_patched()
    from ultralytics import YOLO

    model_tag = Path(model_name).stem.replace('.', '_')
    train_project = project_dir / 'runs_train'
    val_project = project_dir / 'runs_val'
    ensure_dir(train_project)
    ensure_dir(val_project)

    log(f'Training model={model_name} on data={data_yaml}')
    model_path = Path(model_name)
    yaml_variant = model_path.suffix.lower() in {'.yaml', '.yml'}
    if model_path.suffix.lower() in {'.yaml', '.yml'}:
        load_yaml = runtime_model_yaml(model_path, data_yaml, project_dir)
        if load_yaml != model_path:
            log(f'Using runtime model YAML with dataset nc: {load_yaml}')
        model = YOLO(str(load_yaml))
        if args.pretrained and str(args.pretrained).lower() not in {'none', 'false', '0'}:
            log(f'Loading transferable pretrained weights from {args.pretrained} into {model_name}')
            try:
                loaded = model.load(str(args.pretrained))
                if loaded is not None:
                    model = loaded
                copied = copy_detect_head_by_shape(model, args.pretrained)
                if copied:
                    log(f'Copied {copied} Detect-head tensors by shape from {args.pretrained}')
            except Exception as e:
                log(f'WARNING: partial pretrained transfer failed for {model_name}: {e}. Continuing from YAML initialization.')
        init_ckpt = save_initialized_checkpoint(model, project_dir / 'initialized_checkpoints' / f'{model_tag}_init.pt')
        log(f'Training will start from initialized checkpoint: {init_ckpt}')
        model = YOLO(str(init_ckpt))
    else:
        model = YOLO(model_name)
    train_kwargs = {
        'data': str(data_yaml),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'seed': args.seed,
        'project': str(train_project),
        'name': model_tag,
        'exist_ok': True,
        'patience': args.patience,
        'cos_lr': args.cos_lr,
        'plots': True,
    }
    train_results = model.train(**train_kwargs)
    best = train_project / model_tag / 'weights' / 'best.pt'
    if not best.exists():
        # Fallback: locate best.pt inside model directory.
        candidates = list((train_project / model_tag).rglob('best.pt'))
        if candidates:
            best = candidates[0]
        else:
            raise RuntimeError(f'Cannot locate best.pt for {model_name}')

    log(f'Validating best model: {best}')
    trained = YOLO(str(best))
    metrics = trained.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(val_project),
        name=model_tag,
        exist_ok=True,
        plots=True,
        split='test',
        save_json=args.save_json,
    )
    row = metric_from_results(metrics)
    row.update({
        'model': model_tag,
        'weights': str(best),
        'data_yaml': str(data_yaml),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'params': count_params(trained),
        'gflops': estimate_gflops(trained, args.imgsz),
        'model_size_mb': (best.stat().st_size / (1024 * 1024)) if best.exists() else None,
    })
    # Extract Ultralytics timing information when available. Values are ms/image.
    speed = getattr(metrics, 'speed', None)
    if isinstance(speed, dict):
        preprocess = safe_float(speed.get('preprocess'), 0.0) or 0.0
        inference = safe_float(speed.get('inference'), 0.0) or 0.0
        postprocess = safe_float(speed.get('postprocess'), 0.0) or 0.0
        latency = preprocess + inference + postprocess
        row['latency_ms_img'] = latency if latency > 0 else None
        row['fps'] = (1000.0 / latency) if latency and latency > 0 else None
        row['inference_ms_img'] = inference

    if not args.skip_error_counts:
        log(f'Counting fixed-threshold TP/FP/FN for {model_tag}')
        row.update(count_detection_errors(trained, data_yaml, args))

    # Save raw metrics if possible.
    out_json = project_dir / 'metrics' / f'{model_tag}.json'
    ensure_dir(out_json.parent)
    try:
        if hasattr(metrics, 'to_json'):
            raw = json.loads(metrics.to_json())
        elif hasattr(metrics, 'summary'):
            raw = metrics.summary()
        else:
            raw = str(metrics)
    except Exception as e:
        raw = {'repr': str(metrics), 'serialization_error': str(e)}
    save_json(out_json, {'summary': row, 'raw': raw})

    # Copy training results.csv if present.
    src_results = train_project / model_tag / 'results.csv'
    if src_results.exists():
        shutil.copy2(src_results, project_dir / 'metrics' / f'{model_tag}_train_results.csv')
    return row


def validate_subsets(models_rows: List[Dict[str, Any]], subset_yamls: List[Path], project_dir: Path, args) -> List[Dict[str, Any]]:
    ensure_ultralytics_patched()
    from ultralytics import YOLO
    subset_rows = []
    if not subset_yamls:
        return subset_rows
    for row in models_rows:
        weights = row.get('weights')
        if not weights:
            continue
        model_tag = row['model']
        for y in subset_yamls:
            if not y.exists():
                continue
            subset = y.stem.split('_')[-1]
            log(f'Validating subset={subset} with model={model_tag}')
            m = YOLO(weights)
            metrics = m.val(
                data=str(y),
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                project=str(project_dir / 'runs_val_subsets'),
                name=f'{model_tag}_{subset}',
                exist_ok=True,
                plots=True,
                split='test',
            )
            r = metric_from_results(metrics)
            r.update({'model': model_tag, 'subset': subset, 'data_yaml': str(y)})
            if not args.skip_error_counts:
                log(f'Counting fixed-threshold TP/FP/FN for {model_tag}_{subset}')
                r.update(count_detection_errors(m, y, args))
            subset_rows.append(r)
    return subset_rows


def main():
    ap = argparse.ArgumentParser(description='One-click XraySafe-YOLO baseline training/evaluation runner.')
    ap.add_argument('--dataset', choices=['opixray', 'pidray'], required=True)
    ap.add_argument('--raw-root', type=Path, required=True, help='Downloaded raw dataset root.')
    ap.add_argument('--out-root', type=Path, required=True, help='Experiment output root.')
    ap.add_argument('--models', nargs='+', default=['yolo11n.pt', 'yolov8n.pt'], help='Ultralytics model weights or YAML files.')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--symlink', action='store_true', help='Symlink images instead of copying them.')
    ap.add_argument('--skip-prepare', action='store_true')
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--cos-lr', action='store_true')
    ap.add_argument('--save-json', action='store_true')
    ap.add_argument('--pretrained', default='yolo11n.pt', help='Pretrained checkpoint used for YAML variants; use none to train custom YAML from scratch.')
    ap.add_argument('--skip-error-counts', action='store_true', help='Skip fixed-threshold TP/FP/FN recomputation from predictions and labels.')
    ap.add_argument('--count-conf', type=float, default=0.25, help='Confidence threshold for fixed TP/FP/FN counting.')
    ap.add_argument('--count-iou', type=float, default=0.50, help='IoU threshold for fixed TP/FP/FN counting.')
    ap.add_argument('--max-count-images', type=int, default=0, help='Limit images used for fixed TP/FP/FN counting; 0 means all test images.')
    args = ap.parse_args()
    args.raw_root = args.raw_root.resolve()
    args.out_root = args.out_root.resolve()

    ensure_dir(args.out_root)
    prepared_root = args.out_root / 'dataset_yolo'
    if args.skip_prepare:
        data_yaml = prepared_root / f'{args.dataset}.yaml'
        if not data_yaml.exists():
            raise FileNotFoundError(f'--skip-prepare was used, but {data_yaml} does not exist.')
    else:
        prepare = import_prepare(args.dataset)
        data_yaml = prepare(args.raw_root, prepared_root, seed=args.seed, val_ratio=args.val_ratio, symlink=args.symlink)

    save_json(args.out_root / 'run_config.json', vars(args) | {'data_yaml': str(data_yaml)})

    rows = []
    if not args.skip_train:
        for model_name in args.models:
            try:
                rows.append(run_train_and_val(model_name, data_yaml, args.out_root, args))
            except Exception as e:
                log(f'ERROR while running {model_name}: {e}')
                rows.append({'model': Path(model_name).stem, 'error': str(e)})
    else:
        log('Skipping training/evaluation as requested.')

    metrics_dir = ensure_dir(args.out_root / 'metrics')
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(metrics_dir / 'main_comparison.csv', index=False)
        df.to_markdown(metrics_dir / 'main_comparison.md', index=False)
        log(f'Main comparison saved to {metrics_dir / "main_comparison.csv"}')

    subset_yamls = sorted(prepared_root.glob(f'{args.dataset}_*.yaml'))
    subset_rows = validate_subsets(rows, subset_yamls, args.out_root, args) if rows else []
    if subset_rows:
        sdf = pd.DataFrame(subset_rows)
        sdf.to_csv(metrics_dir / 'subset_comparison.csv', index=False)
        sdf.to_markdown(metrics_dir / 'subset_comparison.md', index=False)
        log(f'Subset comparison saved to {metrics_dir / "subset_comparison.csv"}')

    log('Done.')


if __name__ == '__main__':
    main()
