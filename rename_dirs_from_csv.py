import argparse
import csv
import os
import re
import sys

INVALID_CHARS = r'[<>:"/\\|?*\x00-\x1f]'
TMP_PREFIX = '__tmp_rename__'


def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(INVALID_CHARS, '_', name)
    name = re.sub(r'[ \t\r\n]+', ' ', name)
    name = name.rstrip(' .')
    return name or '_'


def load_mapping(csv_path: str, id_column: str, name_column: str, pad: int) -> dict:
    mapping = {}
    seen_names = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if id_column not in reader.fieldnames or name_column not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain columns '{id_column}' and '{name_column}'. Found: {reader.fieldnames}")
        for row in reader:
            raw_id = row[id_column].strip()
            if raw_id == '':
                continue
            try:
                idx = int(float(raw_id))
            except ValueError:
                raise ValueError(f"Invalid numeric id in CSV: '{raw_id}'")
            key = str(idx).zfill(pad)
            label = sanitize_name(row[name_column])
            if not label:
                label = key
            final_name = label
            if final_name in seen_names:
                suffix = seen_names[final_name] + 1
                seen_names[final_name] = suffix
                final_name = f"{label}_{suffix}"
            else:
                seen_names[final_name] = 1
            mapping[key] = final_name
    return mapping


def build_renames(root_dir: str, mapping: dict) -> list:
    dirs = sorted([d for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))])
    renames = []
    for d in dirs:
        if d not in mapping:
            print(f"Warning: directory '{d}' has no entry in the CSV mapping and will be skipped.")
            continue
        new_name = mapping[d]
        if d == new_name:
            continue
        old_path = os.path.join(root_dir, d)
        new_path = os.path.join(root_dir, new_name)
        renames.append((old_path, new_path, d, new_name))
    return renames


def check_conflicts(renames: list, root_dir: str):
    targets = {new for _, new, _, _ in renames}
    if len(targets) != len(renames):
        raise RuntimeError('Duplicate target names found; check the CSV labels.')
    for _, new_path, old_name, new_name in renames:
        if os.path.exists(new_path) and not os.path.isdir(new_path):
            raise RuntimeError(f"Target path exists and is not a directory: {new_path}")
        if os.path.exists(new_path) and os.path.basename(new_path) != os.path.basename(old_name):
            raise RuntimeError(f"Target directory already exists: {new_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description='Rename numbered directories using a CSV label mapping.')
    parser.add_argument('--root', default='data_folder/datasets/phase1/chinese/images',
                        help='Root folder containing numbered directories. Default: %(default)s')
    parser.add_argument('--csv', default='data_folder/datasets/phase1/chinese/food_class.csv',
                        help='CSV file with id/name mapping. Default: %(default)s')
    parser.add_argument('--id-column', default='List No.', help='CSV column for numeric folder IDs. Default: %(default)s')
    parser.add_argument('--name-column', default='English Name', help='CSV column for target folder names. Default: %(default)s')
    parser.add_argument('--pad', type=int, default=3,
                        help='Zero-pad width for numeric folder names (e.g. 000, 001). Default: %(default)s')
    parser.add_argument('--dry-run', action='store_true', help='Show rename operations without executing them.')
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    if not os.path.isdir(root_dir):
        print(f"Error: root folder does not exist: {root_dir}")
        return 1

    mapping = load_mapping(args.csv, args.id_column, args.name_column, args.pad)
    renames = build_renames(root_dir, mapping)
    if not renames:
        print('No directories found to rename.')
        return 0

    print(f"Found {len(renames)} directories to rename in '{root_dir}'.")
    for old_path, new_path, old_name, new_name in renames:
        print(f"{old_name} -> {new_name}")

    if args.dry_run:
        print('\nDry run only; no changes made.')
        return 0

    check_conflicts(renames, root_dir)

    # Use a two-stage rename to avoid collisions with interleaved names
    temp_renames = []
    for old_path, _, old_name, new_name in renames:
        temp_path = os.path.join(root_dir, TMP_PREFIX + os.path.basename(old_path))
        os.rename(old_path, temp_path)
        temp_renames.append((temp_path, os.path.join(root_dir, new_name)))

    for temp_path, final_path in temp_renames:
        os.rename(temp_path, final_path)

    print('\nRename complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
