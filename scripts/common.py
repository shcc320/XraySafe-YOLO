from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def log(msg: str) -> None:
    print(f'[{now()}] {msg}', flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_images(root: Path) -> List[Path]:
    return [p for p in root.rglob('*') if p.suffix.lower() in IMAGE_EXTS]


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # width, height


def normalize_class_name(name: str) -> str:
    return (
        name.strip()
        .replace('-', '_')
        .replace(' ', '_')
        .replace('/', '_')
        .replace('__', '_')
        .lower()
    )


def safe_copy(src: Path, dst: Path, symlink: bool = False) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return
    if symlink:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    # Clamp to valid image boundary.
    x1 = max(0.0, min(float(x1), w - 1.0))
    x2 = max(0.0, min(float(x2), w - 1.0))
    y1 = max(0.0, min(float(y1), h - 1.0))
    y2 = max(0.0, min(float(y2), h - 1.0))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def write_yolo_label(path: Path, rows: Sequence[Tuple[int, float, float, float, float]]) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        for cls, cx, cy, bw, bh in rows:
            if bw <= 0 or bh <= 0:
                continue
            f.write(f'{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n')


def write_yaml(path: Path, data: Dict) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_yaml(path: Path) -> Dict:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)



def json_safe(obj):
    """Convert common non-JSON objects, especially pathlib.Path, before logging configs."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj

def save_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False)


def random_split(items: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * val_ratio))) if items else 0
    return items[n_val:], items[:n_val]


def make_dataset_yaml(out_path: Path, dataset_root: Path, names: Sequence[str], *, train='images/train', val='images/val', test='images/test') -> None:
    data = {
        'path': str(dataset_root.resolve()),
        'train': train,
        'val': val,
        'test': test,
        'names': {i: name for i, name in enumerate(names)},
    }
    write_yaml(out_path, data)


def build_image_index(raw_root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in find_images(raw_root):
        idx[p.name] = p
        idx[p.stem] = p
        # Also keep a lowercase index, useful when annotations differ in case.
        idx[p.name.lower()] = p
        idx[p.stem.lower()] = p
    return idx


def infer_split_from_path(path: Path) -> str:
    parts = [x.lower() for x in path.parts]
    joined = '/'.join(parts)
    if 'val' in parts or 'valid' in parts or 'validation' in parts:
        return 'val'
    if 'train' in parts or 'training' in parts:
        return 'train'
    if 'test' in parts or 'testing' in parts:
        return 'test'
    if any(x in joined for x in ['ol1', 'ol2', 'ol3', 'easy', 'hard', 'hidden']):
        return 'test'
    return 'train'


def is_float_token(token: str) -> bool:
    try:
        float(token)
        return True
    except Exception:
        return False


def parse_flat_annotation_line(line: str) -> Optional[Tuple[str, str, float, float, float, float]]:
    """Parse a generic text row: image category x1 y1 x2 y2.

    The category may contain spaces, so the last four numeric tokens are treated as box coords,
    the first token is the image name, and the middle tokens are joined as the class name.
    """
    line = line.strip().replace(',', ' ')
    if not line or line.startswith('#'):
        return None
    toks = [t for t in line.split() if t]
    if len(toks) < 6:
        return None
    if not all(is_float_token(t) for t in toks[-4:]):
        return None
    image_name = toks[0]
    cls_name = ' '.join(toks[1:-4])
    if not cls_name:
        return None
    x1, y1, x2, y2 = map(float, toks[-4:])
    return image_name, normalize_class_name(cls_name), x1, y1, x2, y2
