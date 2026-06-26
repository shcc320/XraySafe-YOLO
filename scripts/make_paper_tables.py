from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ORDER = [
    'model', 'precision', 'recall', 'f1', 'map50', 'map50_95',
    'false_negative_count', 'fixed_count_precision', 'fixed_count_recall', 'params', 'gflops', 'model_size_mb', 'latency_ms_img', 'fps', 'epochs', 'imgsz', 'batch', 'weights'
]


def fmt_float(x):
    if pd.isna(x):
        return ''
    try:
        return f'{float(x):.4f}'
    except Exception:
        return str(x)


def fmt_int(x):
    if pd.isna(x):
        return ''
    try:
        return str(int(round(float(x))))
    except Exception:
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    main_csv = args.metrics_dir / 'main_comparison.csv'
    if main_csv.exists():
        df = pd.read_csv(main_csv)
        cols = [c for c in ORDER if c in df.columns]
        df = df[cols]
        for c in df.columns:
            if c in ['false_negative_count', 'params', 'epochs', 'imgsz', 'batch']:
                df[c] = df[c].apply(fmt_int)
            elif c not in ['model', 'weights']:
                df[c] = df[c].apply(fmt_float)
        df.to_csv(args.out_dir / 'table_main_comparison.csv', index=False)
        df.to_markdown(args.out_dir / 'table_main_comparison.md', index=False)
        # LaTeX table; still manually inspect before paper submission.
        (args.out_dir / 'table_main_comparison.tex').write_text(
            df.to_latex(index=False, escape=False), encoding='utf-8'
        )

    subset_csv = args.metrics_dir / 'subset_comparison.csv'
    if subset_csv.exists():
        sdf = pd.read_csv(subset_csv)
        cols = [
            c for c in [
                'model', 'subset', 'precision', 'recall', 'f1', 'map50', 'map50_95',
                'false_negative_count', 'fixed_count_precision', 'fixed_count_recall'
            ] if c in sdf.columns
        ]
        sdf = sdf[cols]
        for c in sdf.columns:
            if c == 'false_negative_count':
                sdf[c] = sdf[c].apply(fmt_int)
            elif c not in ['model', 'subset']:
                sdf[c] = sdf[c].apply(fmt_float)
        sdf.to_csv(args.out_dir / 'table_subset_comparison.csv', index=False)
        sdf.to_markdown(args.out_dir / 'table_subset_comparison.md', index=False)
        (args.out_dir / 'table_subset_comparison.tex').write_text(
            sdf.to_latex(index=False, escape=False), encoding='utf-8'
        )


if __name__ == '__main__':
    main()
