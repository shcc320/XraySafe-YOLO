from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable

import pandas as pd

from run_all import count_detection_errors, ensure_ultralytics_patched


COUNT_COLUMNS = [
    'true_positive_count',
    'false_positive_count',
    'false_negative_count',
    'gt_box_count',
    'pred_box_count',
    'fixed_count_precision',
    'fixed_count_recall',
    'count_conf',
    'count_iou',
    'counted_images',
    'count_error_note',
]


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def clean_count_result(result: Dict[str, object]) -> Dict[str, object]:
    cleaned: Dict[str, object] = {}
    for key in COUNT_COLUMNS:
        if key in result:
            cleaned[key] = result[key]
    if 'count_error_note' not in cleaned:
        cleaned['count_error_note'] = ''
    return cleaned


def update_json_summary(metrics_dir: Path, model_name: str, counts: Dict[str, object]) -> None:
    json_path = metrics_dir / f'{model_name}.json'
    if not json_path.exists():
        return
    try:
        payload = json.loads(json_path.read_text(encoding='utf-8'))
    except Exception:
        return
    summary = payload.get('summary')
    if isinstance(summary, dict):
        summary.update(counts)
        json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def recompute_main(metrics_dir: Path, args) -> pd.DataFrame:
    main_csv = metrics_dir / 'main_comparison.csv'
    if not main_csv.exists():
        raise FileNotFoundError(f'Missing {main_csv}')
    df = pd.read_csv(main_csv)
    ensure_ultralytics_patched()
    from ultralytics import YOLO

    counter_args = SimpleNamespace(
        imgsz=args.imgsz,
        batch=args.count_batch,
        count_chunk_size=args.count_chunk_size,
        device=args.device,
        count_conf=args.count_conf,
        count_iou=args.count_iou,
        max_count_images=args.max_count_images,
    )
    for idx, row in df.iterrows():
        weights = row.get('weights')
        model_name = row.get('model')
        data_yaml = Path(row.get('data_yaml') or args.data_yaml)
        if not isinstance(weights, str) or not weights or not Path(weights).exists():
            continue
        print(f'[count] main model={model_name} data={data_yaml}')
        clear_cuda_cache()
        model = YOLO(weights)
        counts = clean_count_result(count_detection_errors(model, data_yaml, counter_args))
        for key, val in counts.items():
            df.loc[idx, key] = val
        if isinstance(model_name, str):
            update_json_summary(metrics_dir, model_name, counts)
        del model
        clear_cuda_cache()
    df.to_csv(main_csv, index=False)
    df.to_markdown(metrics_dir / 'main_comparison.md', index=False)
    return df


def recompute_subsets(metrics_dir: Path, main_df: pd.DataFrame, args) -> None:
    subset_csv = metrics_dir / 'subset_comparison.csv'
    if not subset_csv.exists():
        return
    sdf = pd.read_csv(subset_csv)
    weights_by_model = {
        str(row['model']): str(row['weights'])
        for _, row in main_df.iterrows()
        if 'model' in row and 'weights' in row and isinstance(row.get('weights'), str)
    }
    subset_yaml_by_name = {Path(p).stem.split('_')[-1]: Path(p) for p in args.subset_yamls}

    ensure_ultralytics_patched()
    from ultralytics import YOLO

    counter_args = SimpleNamespace(
        imgsz=args.imgsz,
        batch=args.count_batch,
        count_chunk_size=args.count_chunk_size,
        device=args.device,
        count_conf=args.count_conf,
        count_iou=args.count_iou,
        max_count_images=args.max_count_images,
    )
    for idx, row in sdf.iterrows():
        model_name = str(row.get('model'))
        subset = str(row.get('subset')).lower()
        weights = weights_by_model.get(model_name)
        data_yaml = subset_yaml_by_name.get(subset)
        if not weights or not Path(weights).exists() or data_yaml is None:
            continue
        print(f'[count] subset model={model_name} subset={subset} data={data_yaml}')
        clear_cuda_cache()
        model = YOLO(weights)
        counts = clean_count_result(count_detection_errors(model, data_yaml, counter_args))
        for key, val in counts.items():
            sdf.loc[idx, key] = val
        del model
        clear_cuda_cache()
    sdf.to_csv(subset_csv, index=False)
    sdf.to_markdown(metrics_dir / 'subset_comparison.md', index=False)


def existing_subset_yamls(data_yaml: Path) -> Iterable[Path]:
    root = data_yaml.parent
    stem = data_yaml.stem
    return sorted(root.glob(f'{stem}_*.yaml'))


def main() -> None:
    ap = argparse.ArgumentParser(description='Recompute fixed-threshold TP/FP/FN counts from saved best.pt weights.')
    ap.add_argument('--metrics-dir', type=Path, required=True)
    ap.add_argument('--data-yaml', type=Path, required=True)
    ap.add_argument('--subset-yamls', type=Path, nargs='*')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0')
    ap.add_argument('--count-batch', type=int, default=1)
    ap.add_argument('--count-chunk-size', type=int, default=32)
    ap.add_argument('--count-conf', type=float, default=0.25)
    ap.add_argument('--count-iou', type=float, default=0.50)
    ap.add_argument('--max-count-images', type=int, default=0)
    args = ap.parse_args()
    args.metrics_dir = args.metrics_dir.resolve()
    args.data_yaml = args.data_yaml.resolve()
    if args.subset_yamls:
        args.subset_yamls = [p.resolve() for p in args.subset_yamls]
    else:
        args.subset_yamls = list(existing_subset_yamls(args.data_yaml))

    main_df = recompute_main(args.metrics_dir, args)
    recompute_subsets(args.metrics_dir, main_df, args)
    print('[count] done')


if __name__ == '__main__':
    main()
