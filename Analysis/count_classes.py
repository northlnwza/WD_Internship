"""Count YOLOseg instances per class for any label folder.

Each non-empty line in a .txt label file = one instance; first token = class id (0-based).
Class names come from a class-names file (line 1 = class id 0, ...).

Usage:
    python count_classes.py                      # pops up a folder picker
    python count_classes.py path/to/labels_dir   # use given folder, no dialog
    python count_classes.py labels_dir classes.txt
"""
from pathlib import Path
from collections import Counter
import sys
import pandas as pd


def pick_folder():
    """Open a directory-picker dialog; return the chosen Path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    chosen = filedialog.askdirectory(title='Select label folder (.txt files)')
    root.destroy()
    return Path(chosen) if chosen else None


def find_class_file(label_dir):
    """Look for a class-names file near the label folder."""
    for parent in (label_dir, *label_dir.parents):
        for name in ('6class.txt', 'classes.txt'):
            cand = parent / name
            if cand.is_file():
                return cand
    return None


def main(argv):
    # --- resolve label folder ---
    if len(argv) >= 1:
        label_dir = Path(argv[0]).expanduser().resolve()
    else:
        label_dir = pick_folder()
        if label_dir is None:
            print('No folder selected.')
            return 1
    if not label_dir.is_dir():
        print(f'Not a folder: {label_dir}')
        return 1

    # --- resolve class-names file ---
    class_file = Path(argv[1]).expanduser().resolve() if len(argv) >= 2 else find_class_file(label_dir)
    id2name = {}
    if class_file and class_file.is_file():
        names = [l.strip() for l in class_file.read_text().splitlines() if l.strip()]
        id2name = {i: n for i, n in enumerate(names)}
    else:
        print('No class-names file found; using numeric class ids.')

    # --- count instances per class ---
    counts = Counter()
    files = empty = 0
    for txt in label_dir.glob('*.txt'):
        files += 1
        lines = [ln for ln in txt.read_text().splitlines() if ln.strip()]
        if not lines:
            empty += 1
        for ln in lines:
            counts[int(ln.split()[0])] += 1

    if files == 0:
        print(f'No .txt label files in {label_dir}')
        return 1

    # --- build table ---
    ids = sorted(set(id2name) | set(counts))
    rows = [{
        'class_id': cid,
        'class_name': id2name.get(cid, f'UNKNOWN_{cid}'),
        'count': counts.get(cid, 0),
    } for cid in ids]
    df = pd.DataFrame(rows)
    total = df['count'].sum()
    df['pct'] = (df['count'] / total * 100).round(2) if total else 0.0

    # --- report ---
    print(f'\nfolder: {label_dir}')
    print(f'label files: {files}  (empty: {empty})')
    print(f'total instances: {total}\n')
    print(df.to_string(index=False))

    # --- save outputs next to the label folder, named after it ---
    stem = f'class_counts_{label_dir.name}'
    out_csv = label_dir.parent / f'{stem}.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nsaved: {out_csv}')

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df['class_name'], df['count'], color='steelblue')
        ax.set_ylabel('instances')
        ax.set_title(f'Class item count — {label_dir.name}')
        for i, v in enumerate(df['count']):
            ax.text(i, v, str(v), ha='center', va='bottom')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        out_png = label_dir.parent / f'{stem}.png'
        plt.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f'saved: {out_png}')
    except Exception as e:
        print(f'(skipped chart: {e})')

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
