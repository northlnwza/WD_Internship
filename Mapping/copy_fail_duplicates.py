r"""
For each day, list the serial numbers (SNs) that appear in the Fail folder, then
copy every Image-folder picture with one of those SNs into that day's fail_copy.

Layout (the two trees mirror each other):
    Fail \<Station>\<YYYYMMDD>\...\<file>.jpg
    Image\<Station>\<YYYYMMDD>\...\<file>.jpg

A file's SN is the first underscore-delimited token of its name:
    Fail  name:  ZQHAJMSE_20260618092445_ST5_TOP_EC15_Fail.jpg  -> SN = ZQHAJMSE
    Image name:  WBGRNHTP_20260622194626_ST5_TOP    _EC1.jpg        -> SN = WBGRNHTP

Only the SN is used for matching. The copied files come from the Image tree
(names without "_Fail") and are placed, keeping their Image name, into:
    Fail\<Station>\<YYYYMMDD>\failed_raw
"""

import argparse
import re
import shutil
from pathlib import Path


def sn_of(path: Path) -> str:
    """Serial number = first underscore-delimited token of the file name."""
    return path.name.split("_", 1)[0]


def process_station(root: Path, station: str, dry_run: bool) -> int:
    fail_station = root / "Fail" / station
    image_station = root / "Image" / station

    if not fail_station.is_dir():
        raise SystemExit(f"Fail station folder not found: {fail_station}")

    total_copied = 0
    day_pattern = re.compile(r"^\d{8}$")  # YYYYMMDD

    for day_dir in sorted(p for p in fail_station.iterdir() if p.is_dir()):
        if not day_pattern.match(day_dir.name):
            continue  # skip non-date folders such as ZZZZ

        # 1. All SNs that appear in the Fail folder for this day.
        fail_sns = {sn_of(f) for f in day_dir.rglob("*.jpg")}
        if not fail_sns:
            print(f"Day {day_dir.name}: 0 Fail SNs, skipped.")
            continue

        # 2. Image pictures for this day whose SN is in the Fail list.
        img_day = image_station / day_dir.name
        if not img_day.is_dir():
            print(f"Day {day_dir.name}: {len(fail_sns)} Fail SNs, "
                  f"no Image folder, 0 copied.")
            continue

        img_pics = [
            f for f in img_day.rglob("*.jpg")
            if "fail_copy" not in f.parts          # don't re-scan our own output
        ]
        matches = [f for f in img_pics if sn_of(f) in fail_sns]

        dest_root = day_dir / "failed_raw"
        copied = 0
        for f in matches:
            rel = f.relative_to(img_day)
            dest = dest_root / rel
            if dry_run:
                print(f"[would copy] {f} -> {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    dest.unlink()
                shutil.copy2(f, dest)
                print(f"  copied {rel}  (SN {sn_of(f)})")
            copied += 1

        total_copied += copied
        verb = "would copy" if dry_run else "copied"
        print(f"Day {day_dir.name}: {len(fail_sns)} Fail SNs, "
              f"{len(img_pics)} Image pictures, {copied} {verb}.")

    return total_copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy Image pictures of failed SNs into each day's fail_copy folder."
    )
    parser.add_argument("--station", default="ST5_SEW-TOP",
                        help="Station sub-folder name (default: ST5_SEW-TOP).")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent),
                        help="Project root containing Fail\\ and Image\\ (default: script folder).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be copied without writing anything.")
    args = parser.parse_args()

    total = process_station(Path(args.root), args.station, args.dry_run)

    verb = "would be" if args.dry_run else "were"
    print()
    print(f"Done. {total} image(s) {verb} copied for station '{args.station}'.")


if __name__ == "__main__":
    main()
