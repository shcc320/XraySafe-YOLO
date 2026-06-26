from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter
import yaml


def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def count_split(root: Path, split_path: str):
    img_dir = root / split_path
    label_dir = root / split_path.replace('images', 'labels', 1)
    imgs = [p for p in img_dir.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}] if img_dir.exists() else []
    labels = [p for p in label_dir.rglob('*.txt')] if label_dir.exists() else []
    cls = Counter()
    boxes = 0
    for lb in labels:
        for line in lb.read_text(encoding='utf-8', errors='ignore').splitlines():
            toks = line.strip().split()
            if len(toks) >= 5:
                cls[int(float(toks[0]))] += 1
                boxes += 1
    return len(imgs), len(labels), boxes, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    args = ap.parse_args()
    data = read_yaml(args.data)
    root = Path(data['path'])
    names = data['names']
    if isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    else:
        names = {i: v for i, v in enumerate(names)}
    print(f'Dataset root: {root}')
    for split in ['train', 'val', 'test']:
        if split in data:
            n_img, n_lab, boxes, cls = count_split(root, data[split])
            print(f'[{split}] images={n_img}, label_files={n_lab}, boxes={boxes}')
            for cid, c in sorted(cls.items()):
                print(f'  - {cid}:{names.get(cid,cid)} -> {c}')


if __name__ == '__main__':
    main()
