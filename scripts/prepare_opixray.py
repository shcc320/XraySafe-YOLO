from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from common import (
    build_image_index,
    ensure_dir,
    find_images,
    image_size,
    infer_split_from_path,
    log,
    make_dataset_yaml,
    normalize_class_name,
    parse_flat_annotation_line,
    random_split,
    safe_copy,
    save_json,
    write_yolo_label,
    xyxy_to_yolo,
)

OPIXRAY_NAMES = [
    'folding_knife',
    'straight_knife',
    'scissor',
    'utility_knife',
    'multi_tool_knife',
]
ALIASES = {
    'foldingknife': 'folding_knife',
    'folding_knife': 'folding_knife',
    'folding': 'folding_knife',
    'straightknife': 'straight_knife',
    'straight_knife': 'straight_knife',
    'straight': 'straight_knife',
    'scissor': 'scissor',
    'scissors': 'scissor',
    'utilityknife': 'utility_knife',
    'utility_knife': 'utility_knife',
    'utility': 'utility_knife',
    'multi_tool_knife': 'multi_tool_knife',
    'multitoolknife': 'multi_tool_knife',
    'multi_tool': 'multi_tool_knife',
    'multi': 'multi_tool_knife',
}


def canonical_cls(name: str) -> str:
    n = normalize_class_name(name).replace('_', '')
    n2 = normalize_class_name(name)
    return ALIASES.get(n2, ALIASES.get(n, n2))




def _tokens_have_last4_numbers(toks: List[str]) -> bool:
    if len(toks) < 5:
        return False
    try:
        [float(x) for x in toks[-4:]]
        return True
    except Exception:
        return False


def _parse_official_ann_line(line: str) -> List[Tuple[str, float, float, float, float]]:
    """Parse one OPIXray official annotation row.

    Official OPIXray annotations are stored per image under train_annotation/
    and test_annotation/. The original reader treats tokens 1..5 as:
    class_name, x1, y1, x2, y2, while token 0 may be an image id/index.
    Some mirrors also store rows as: class_name x1 y1 x2 y2.
    This parser accepts both forms.
    """
    line = line.strip().replace(',', ' ')
    if not line or line.startswith('#'):
        return []
    toks = [t for t in line.split() if t]
    if not _tokens_have_last4_numbers(toks):
        return []
    x1, y1, x2, y2 = map(float, toks[-4:])
    candidates: List[str] = []
    # Common official form: <unused_or_image_id> <class> x1 y1 x2 y2
    if len(toks) >= 6:
        candidates.append(' '.join(toks[1:-4]))
    # Mirror form: <class> x1 y1 x2 y2
    candidates.append(' '.join(toks[:-4]))
    # Last resort for single-token class before coords
    if len(toks) >= 5:
        candidates.append(toks[-5])
    for cls in candidates:
        c = canonical_cls(cls)
        if c in OPIXRAY_NAMES:
            return [(c, x1, y1, x2, y2)]
    return [(canonical_cls(candidates[0] if candidates else ''), x1, y1, x2, y2)]


def _candidate_split_roots(raw_root: Path, split: str) -> List[Path]:
    return [
        raw_root / split,
        raw_root / split.capitalize(),
        raw_root,
    ]


def _find_official_split_dirs(raw_root: Path, split: str):
    """Return (split_root, image_dir, ann_dir, list_files) for official OPIXray layout."""
    img_names = [f'{split}_image', f'{split}_images', 'image', 'images']
    ann_names = [f'{split}_annotation', f'{split}_annotations', 'annotation', 'annotations']
    for split_root in _candidate_split_roots(raw_root, split):
        if not split_root.exists():
            continue
        image_dir = next((split_root / x for x in img_names if (split_root / x).exists()), None)
        ann_dir = next((split_root / x for x in ann_names if (split_root / x).exists()), None)
        list_files = sorted(split_root.glob(f'{split}_knife*.txt')) + sorted(split_root.glob('*knife*.txt'))
        # Keep stable unique order.
        uniq = []
        seen = set()
        for lf in list_files:
            if lf.resolve() not in seen:
                uniq.append(lf)
                seen.add(lf.resolve())
        list_files = uniq
        if image_dir and ann_dir and list_files:
            return split_root, image_dir, ann_dir, list_files
    return None


def _read_ids(list_file: Path) -> List[str]:
    ids = []
    for line in list_file.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Official list files contain one image id per row. If a mirror has extra
        # columns, keep the first token as the id.
        ids.append(line.split()[0])
    return ids


def _find_image_by_id(image_dir: Path, image_id: str) -> Path | None:
    iid = Path(image_id).stem
    # Try exact and common OPIXray extensions first.
    for ext in ['', '.TIFF', '.tiff', '.TIF', '.tif', '.jpg', '.jpeg', '.png', '.bmp']:
        p = image_dir / (iid + ext)
        if p.exists() and p.is_file():
            return p
    # Fallback recursive search by stem.
    for p in image_dir.rglob('*'):
        if p.is_file() and p.stem == iid and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}:
            return p
    return None


def _find_ann_by_id(ann_dir: Path, image_id: str) -> Path | None:
    iid = Path(image_id).stem
    for ext in ['.txt', '.TXT']:
        p = ann_dir / (iid + ext)
        if p.exists() and p.is_file():
            return p
    for p in ann_dir.rglob('*.txt'):
        if p.stem == iid:
            return p
    return None


def convert_official_opixray(raw_root: Path, out_root: Path, seed: int, val_ratio: float, symlink: bool) -> Path | None:
    """Convert official OPIXray layout to YOLO.

    Supported layout:
      OPIXray/
        train/train_image, train/train_annotation, train/train_knife.txt
        test/test_image,  test/test_annotation,  test/test_knife.txt,
                                      test/test_knife-1.txt, -2, -3
    """
    train_info = _find_official_split_dirs(raw_root, 'train')
    test_info = _find_official_split_dirs(raw_root, 'test')
    if not train_info and not test_info:
        return None

    log('Detected official OPIXray layout with train_image/test_image and annotation folders.')
    cls_to_id = {c: i for i, c in enumerate(OPIXRAY_NAMES)}
    rows_by_image: Dict[Path, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    images_by_split: Dict[str, List[Path]] = defaultdict(list)
    missing_images: List[str] = []
    missing_annotations: List[str] = []
    unknown_classes = set()

    def process_primary(split: str, info):
        if not info:
            return
        _split_root, image_dir, ann_dir, list_files = info
        primary = next((x for x in list_files if x.name.lower() == f'{split}_knife.txt'), list_files[0])
        ids = _read_ids(primary)
        log(f'Using {primary} with {len(ids)} image ids for split={split}')
        for image_id in ids:
            img_path = _find_image_by_id(image_dir, image_id)
            if img_path is None:
                missing_images.append(f'{split}:{image_id}')
                continue
            ann_path = _find_ann_by_id(ann_dir, image_id)
            if ann_path is None:
                missing_annotations.append(f'{split}:{image_id}')
                parsed = []
            else:
                parsed = []
                for line in ann_path.read_text(encoding='utf-8', errors='ignore').splitlines():
                    parsed.extend(_parse_official_ann_line(line))
            w, h = image_size(img_path)
            for cls, x1, y1, x2, y2 in parsed:
                cls = canonical_cls(cls)
                if cls not in cls_to_id:
                    unknown_classes.add(cls)
                    continue
                cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
                rows_by_image[img_path].append((cls_to_id[cls], cx, cy, bw, bh))
            images_by_split[split].append(img_path)

    process_primary('train', train_info)
    process_primary('test', test_info)

    # Deterministic validation split from official train set.
    if images_by_split.get('train') and not images_by_split.get('val'):
        train, val = random_split(images_by_split['train'], val_ratio, seed)
        images_by_split['train'] = train
        images_by_split['val'] = val

    for split, imgs in images_by_split.items():
        log(f'Writing {len(imgs)} images for split={split}')
        for img_path in imgs:
            safe_copy(img_path, out_root / 'images' / split / img_path.name, symlink=symlink)
            write_yolo_label(out_root / 'labels' / split / f'{img_path.stem}.txt', rows_by_image.get(img_path, []))

    subset_info = {}
    if test_info:
        _split_root, image_dir, ann_dir, list_files = test_info
        subset_map = {'1': 'OL1', '2': 'OL2', '3': 'OL3'}
        for key, subname in subset_map.items():
            lf = next((x for x in list_files if f'test_knife-{key}' in x.name.lower() or f'knife-{key}' in x.name.lower()), None)
            if not lf:
                continue
            ids = _read_ids(lf)
            log(f'Using {lf} with {len(ids)} image ids for subset={subname}')
            written = 0
            for image_id in ids:
                img_path = _find_image_by_id(image_dir, image_id)
                if img_path is None:
                    continue
                # Ensure row cache exists, even if subset image did not appear in test_knife.txt.
                if img_path not in rows_by_image:
                    ann_path = _find_ann_by_id(ann_dir, image_id)
                    if ann_path:
                        w, h = image_size(img_path)
                        for line in ann_path.read_text(encoding='utf-8', errors='ignore').splitlines():
                            for cls, x1, y1, x2, y2 in _parse_official_ann_line(line):
                                cls = canonical_cls(cls)
                                if cls in cls_to_id:
                                    cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
                                    rows_by_image[img_path].append((cls_to_id[cls], cx, cy, bw, bh))
                safe_copy(img_path, out_root / 'images' / subname / img_path.name, symlink=symlink)
                write_yolo_label(out_root / 'labels' / subname / f'{img_path.stem}.txt', rows_by_image.get(img_path, []))
                written += 1
            if written:
                y = out_root / f'opixray_{subname.lower()}.yaml'
                make_dataset_yaml(y, out_root, OPIXRAY_NAMES, test=f'images/{subname}')
                subset_info[subname.lower()] = str(y)

    if unknown_classes:
        log(f'WARNING: unknown classes skipped: {sorted(unknown_classes)}')
    if missing_images:
        log(f'WARNING: {len(missing_images)} listed images were not found under official image dirs.')
    if missing_annotations:
        log(f'WARNING: {len(missing_annotations)} listed annotation files were not found.')

    yaml_path = out_root / 'opixray.yaml'
    make_dataset_yaml(yaml_path, out_root, OPIXRAY_NAMES)
    save_json(out_root / 'conversion_report.json', {
        'format': 'official_opixray',
        'raw_root': str(raw_root),
        'out_root': str(out_root),
        'names': OPIXRY_NAMES if False else OPIXRAY_NAMES,
        'splits': {k: len(v) for k, v in images_by_split.items()},
        'subset_yamls': subset_info,
        'unknown_classes': sorted(unknown_classes),
        'missing_images_count': len(missing_images),
        'missing_annotations_count': len(missing_annotations),
    })
    log(f'Dataset YAML written to {yaml_path}')
    return yaml_path


def collect_flat_annotations(raw_root: Path) -> Dict[str, List[Tuple[str, float, float, float, float]]]:
    """Collect annotations from flat txt files with rows: image category x1 y1 x2 y2."""
    ann: Dict[str, List[Tuple[str, float, float, float, float]]] = defaultdict(list)
    for txt in raw_root.rglob('*.txt'):
        # Skip YOLO labels if present: rows look like cls cx cy w h.
        try:
            lines = txt.read_text(encoding='utf-8', errors='ignore').splitlines()
        except Exception:
            continue
        parsed_count = 0
        for line in lines:
            row = parse_flat_annotation_line(line)
            if row is None:
                continue
            img, cls, x1, y1, x2, y2 = row
            cls = canonical_cls(cls)
            ann[img].append((cls, x1, y1, x2, y2))
            parsed_count += 1
        if parsed_count:
            log(f'Parsed {parsed_count} flat annotations from {txt}')
    return ann


def copy_existing_yolo(raw_root: Path, out_root: Path, symlink: bool) -> bool:
    """If raw_root already has YOLO folders, copy/symlink them directly."""
    if not (raw_root / 'images').exists() or not (raw_root / 'labels').exists():
        return False
    log('Detected existing YOLO-format dataset. Copying/symlinking images and labels...')
    for sub in ['train', 'val', 'test']:
        for kind in ['images', 'labels']:
            src_dir = raw_root / kind / sub
            if not src_dir.exists():
                continue
            for p in src_dir.rglob('*'):
                if p.is_file():
                    safe_copy(p, out_root / kind / sub / p.name, symlink=symlink)
    return True



def _move_matching_label(img_path: Path, src_split: str, dst_split: str, out_root: Path) -> None:
    dst_img = out_root / 'images' / dst_split / img_path.name
    safe_copy(img_path, dst_img, symlink=False)
    try:
        img_path.unlink()
    except Exception:
        pass
    src_label = out_root / 'labels' / src_split / f'{img_path.stem}.txt'
    if src_label.exists():
        dst_label = out_root / 'labels' / dst_split / src_label.name
        safe_copy(src_label, dst_label, symlink=False)
        try:
            src_label.unlink()
        except Exception:
            pass


def ensure_existing_yolo_splits(out_root: Path, seed: int, val_ratio: float) -> None:
    """Create val/test splits from train when an already-converted YOLO dataset lacks them.

    Many OPIXray/PIDray conversions provide only train/test. Ultralytics requires a valid
    validation path during training, so we make a deterministic validation subset.
    """
    train_dir = out_root / 'images' / 'train'
    if not train_dir.exists():
        return
    train_imgs = sorted([p for p in train_dir.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}])
    if not train_imgs:
        return
    import random
    rng = random.Random(seed)
    if not any((out_root / 'images' / 'val').rglob('*')) if (out_root / 'images' / 'val').exists() else True:
        candidates = list(train_imgs)
        rng.shuffle(candidates)
        n_val = int(round(len(candidates) * val_ratio))
        if len(candidates) > 1:
            n_val = max(1, min(n_val, len(candidates) - 1))
        else:
            n_val = 0
        for img in candidates[:n_val]:
            _move_matching_label(img, 'train', 'val', out_root)
        train_imgs = sorted([p for p in train_dir.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}])
    if not any((out_root / 'images' / 'test').rglob('*')) if (out_root / 'images' / 'test').exists() else True:
        candidates = list(train_imgs)
        rng.shuffle(candidates)
        n_test = int(round(len(candidates) * val_ratio))
        if len(candidates) > 1:
            n_test = max(1, min(n_test, len(candidates) - 1))
        else:
            n_test = 0
        for img in candidates[:n_test]:
            _move_matching_label(img, 'train', 'test', out_root)

def prepare(raw_root: Path, out_root: Path, seed: int = 42, val_ratio: float = 0.1, symlink: bool = False) -> Path:
    raw_root = raw_root.resolve()
    out_root = out_root.resolve()
    ensure_dir(out_root)

    official_yaml = convert_official_opixray(raw_root, out_root, seed=seed, val_ratio=val_ratio, symlink=symlink)
    if official_yaml is not None:
        return official_yaml

    if copy_existing_yolo(raw_root, out_root, symlink=symlink):
        ensure_existing_yolo_splits(out_root, seed, val_ratio)
        yaml_path = out_root / 'opixray.yaml'
        make_dataset_yaml(yaml_path, out_root, OPIXRAY_NAMES)
        return yaml_path

    image_idx = build_image_index(raw_root)
    if not image_idx:
        raise RuntimeError(f'No images found under {raw_root}')

    flat_ann = collect_flat_annotations(raw_root)
    if not flat_ann:
        raise RuntimeError(
            'No supported OPIXray annotations were found. Expected flat text rows like: '
            'image_name category x1 y1 x2 y2, or an already converted YOLO dataset with images/ and labels/ folders.'
        )

    # Convert annotations.
    cls_to_id = {c: i for i, c in enumerate(OPIXRAY_NAMES)}
    images_by_split: Dict[str, List[Path]] = defaultdict(list)
    rows_by_image: Dict[Path, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    unknown_classes = set()
    missing_images = []

    for img_key, boxes in flat_ann.items():
        img_path = image_idx.get(img_key) or image_idx.get(Path(img_key).name) or image_idx.get(Path(img_key).stem)
        if img_path is None:
            missing_images.append(img_key)
            continue
        w, h = image_size(img_path)
        for cls, x1, y1, x2, y2 in boxes:
            cls = canonical_cls(cls)
            if cls not in cls_to_id:
                unknown_classes.add(cls)
                continue
            cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            rows_by_image[img_path].append((cls_to_id[cls], cx, cy, bw, bh))

    if unknown_classes:
        log(f'WARNING: unknown classes skipped: {sorted(unknown_classes)}')
    if missing_images:
        log(f'WARNING: {len(missing_images)} annotation image names were not found under raw root.')

    for img_path in rows_by_image:
        split = infer_split_from_path(img_path)
        images_by_split[split].append(img_path)

    # If no validation split exists, split from train.
    if not images_by_split.get('val'):
        train, val = random_split(images_by_split.get('train', []), val_ratio, seed)
        images_by_split['train'] = train
        images_by_split['val'] = val

    # If no test split detected, split from train again, but warn.
    if not images_by_split.get('test'):
        log('WARNING: no test split detected. Creating a small test split from training images for smoke testing only.')
        train, test = random_split(images_by_split.get('train', []), val_ratio, seed + 1)
        images_by_split['train'] = train
        images_by_split['test'] = test

    for split, imgs in images_by_split.items():
        log(f'Writing {len(imgs)} images for split={split}')
        for img_path in imgs:
            safe_copy(img_path, out_root / 'images' / split / img_path.name, symlink=symlink)
            write_yolo_label(out_root / 'labels' / split / f'{img_path.stem}.txt', rows_by_image.get(img_path, []))

    # Make optional subset yamls for OL1/OL2/OL3 if such directories/files are detected.
    subset_info = {}
    for subset in ['ol1', 'ol2', 'ol3']:
        subset_imgs = [p for p in rows_by_image if subset in '/'.join(x.lower() for x in p.parts)]
        if subset_imgs:
            subname = subset.upper()
            for img_path in subset_imgs:
                safe_copy(img_path, out_root / 'images' / subname / img_path.name, symlink=symlink)
                write_yolo_label(out_root / 'labels' / subname / f'{img_path.stem}.txt', rows_by_image.get(img_path, []))
            y = out_root / f'opixray_{subset}.yaml'
            make_dataset_yaml(y, out_root, OPIXRAY_NAMES, test=f'images/{subname}')
            subset_info[subset] = str(y)

    yaml_path = out_root / 'opixray.yaml'
    make_dataset_yaml(yaml_path, out_root, OPIXRAY_NAMES)
    save_json(out_root / 'conversion_report.json', {
        'raw_root': str(raw_root),
        'out_root': str(out_root),
        'names': OPIXRAY_NAMES,
        'splits': {k: len(v) for k, v in images_by_split.items()},
        'subset_yamls': subset_info,
        'unknown_classes': sorted(unknown_classes),
        'missing_images_count': len(missing_images),
    })
    log(f'Dataset YAML written to {yaml_path}')
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', required=True, type=Path)
    ap.add_argument('--out-root', required=True, type=Path)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--symlink', action='store_true')
    args = ap.parse_args()
    prepare(args.raw_root, args.out_root, args.seed, args.val_ratio, args.symlink)


if __name__ == '__main__':
    main()
