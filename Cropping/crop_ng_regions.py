"""
Crop a tight RAW image region around each detected defect, for use as model input.

Context
-------
On each RESULT image the inspection machine draws user-defined inspection regions:
  - GREEN box : region inspected, no defect found (OK)
  - RED box   : region inspected, a defect WAS found (NG)
  - ORANGE box: the AVI's exact detected-defect area, nested inside the red box.

We only crop NG (red) regions - that is where a real defect was found. Inside each red
box we recover the tight ORANGE defect box(es): orange pixels (hue 5-18) with the gold
flex-cable region excluded (the cable shares the orange hue; its mask hue is kept tight
so it doesn't flood the green-gold PCB and swallow real boxes near the board edge). A
zone may hold several non-overlapping defects, so we keep ALL recovered boxes, not the
largest only (e.g. ZCH31KWB's two; ZQHAY7SH's lone box hard against the board's far edge).

The red outline anti-aliases into the orange hue, so its edges leak into the orange mask
- as a thin strip hugging the border, or, in a thin zone, merging with the real box and
ballooning it to the whole zone. So the search runs with a PERIM_FRAME-px border blanked
to peel that bleed off (real boxes are nested/inset and survive). A thin edge-hugging
sliver is kept only when a real box also anchors the zone (then it's a plausible second
defect); a sliver alone is bleed, so we discard it and retry the full zone for a thin-zone
box that fills the border, else fall back to the whole zone (see find_defect_boxes /
_is_border_sliver). This yields tight, defect-focused crops instead of the whole zone. If no
orange box can be recovered at all (e.g. faint diffuse contamination that the machine
drew no box around), we fall back to cropping the whole red zone.

For each defect box we crop a fixed CROP_SIZE × CROP_SIZE window from the full-resolution
RAW image, centred on the defect centroid. If the window would extend beyond the raw image
boundary it is shifted inward along that axis so it stays exactly CROP_SIZE × CROP_SIZE
(unless the raw image itself is smaller than CROP_SIZE in that dimension). This gives every
crop a uniform size for YOLO training while keeping the defect centered.
Save the crop and a side-by-side validation figure:

    [ downscaled RAW with a BLUE rectangle marking the crop border ] | [ the crop ]

Outputs (under debug_out/):
    crop/{station}-{view}/{serial}_{date}_{station}-{view}_{i}.png   the saved RAW crop (the "data")
    sidebyside/{serial}_{date}_{zone}-{view}_{i}.png   raw-with-blue-border | crop
    diag/{serial}_detection.png     downscaled result + legend, with the box this script crops
"""

import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless: save figures to files
import matplotlib.pyplot as plt

# ---- Config -----------------------------------------------------------------
RAW_IMAGE_DIR    = "./images/raw"
RESULT_IMAGE_DIR = "./images/result"
OUT_CROP_BASE    = "./crop"
OUT_SIDE_DIR     = "./debug_out/sidebyside"
OUT_DIAG_DIR     = "./debug_out/diag"
IMAGE_EXT        = ".jpg"
FAIL_STATUS      = "Fail"

CROP_SIZE        = 640          # fixed crop window (RAW pixels); centred on defect centroid

# Red NG-box HSV. Red wraps hue 0/180; floored away from orange (hue >=5) so the
# AVI's orange defect boxes are not mistaken for red user boxes.
RED_RANGES = [((0, 90, 90), (4, 255, 255)), ((176, 90, 90), (180, 255, 255))]

MIN_BOX_AREA     = 300          # ignore small red noise / PCB speckle
MIN_BOX_W        = 12
MIN_BOX_H        = 10
CLOSE_KERNEL     = 15           # rejoin a red outline broken by crossing green lines
MERGE_GAP        = 20           # merge red boxes whose bounds are within this many px

# Orange defect-box recovery (inside each red NG box)
ORANGE_LOWER     = (5, 90, 90)
ORANGE_UPPER     = (18, 255, 255)
# Gold flex cable shares the orange hue; mask + exclude it so its edges aren't recovered.
# Upper hue is held at 35 (metallic gold), NOT up into the yellow-green of the PCB: a
# wider range floods the whole PCBA board into the mask and its dilation then swallows
# real orange boxes near it (e.g. ZQHAY7SH's box hard against the board's right edge).
CABLE_LOWER      = (18, 50, 50)
CABLE_UPPER      = (35, 255, 255)
CABLE_MIN_AREA   = 1500         # only large gold blobs are the cable (not small marks)
CABLE_DILATE     = 21           # margin around the cable to drop its bleeding edges
DEFECT_CLOSE     = 7            # rejoin the orange outline into one blob
DEFECT_MIN_AREA  = 400          # smallest orange contour accepted as a defect box
DEFECT_MIN_WH    = 10
# We keep ALL defect boxes that survive (a zone can hold several non-overlapping ones),
# not just the largest -- so multi-defect zones are captured in full.

# Border-bleed handling. The RED zone outline anti-aliases into the orange hue, so its
# edges leak into the orange mask: as a thin strip hugging the border (a false "defect"),
# or, in a thin zone, merging with the real nested box so the box balloons to the whole
# red zone. Primary defence: blank a PERIM_FRAME-px frame at the red-zone border before
# searching, peeling the outline bleed off so real (inset) boxes stand alone. If framing
# removes everything (a real box that fills a thin zone touches the border), retry on the
# unframed zone and drop only pure border slivers via _is_border_sliver.
PERIM_FRAME      = 4            # px frame blanked at the red-zone border (the outline bleed)
SLIVER_EDGE_TOL  = 1            # px: "flush" against frame inner edge (PERIM_FRAME already blanked the 4px bleed)
SLIVER_FRAC      = 0.50         # sliver spans < this fraction of the perpendicular zone dim
SLIVER_ASPECT    = 4.0          # sliver is at least this elongated (long/short side)


def _is_border_sliver(lx, ly, w, h, rw, rh):
    """True if a candidate box (local to its red zone) is just the red outline's
    orange-hued bleed: flush to an edge, thin across the zone, and elongated."""
    if max(w, h) / max(1, min(w, h)) < SLIVER_ASPECT:
        return False
    touch_h = ly <= SLIVER_EDGE_TOL or (ly + h) >= rh - SLIVER_EDGE_TOL     # top/bottom edge
    touch_v = lx <= SLIVER_EDGE_TOL or (lx + w) >= rw - SLIVER_EDGE_TOL     # left/right edge
    return (touch_h and h < SLIVER_FRAC * rh) or (touch_v and w < SLIVER_FRAC * rw)


def red_mask(hsv):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RED_RANGES:
        m = cv2.bitwise_or(m, cv2.inRange(hsv, np.array(lo), np.array(hi)))
    return m


def _merge_boxes(boxes, gap=MERGE_GAP):
    """Union-merge boxes whose (gap-expanded) rectangles overlap."""
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            x, y, w, h = boxes.pop()
            ax0, ay0, ax1, ay1 = x - gap, y - gap, x + w + gap, y + h + gap
            merged = True
            while merged:
                merged = False
                rest = []
                for (bx, by, bw, bh) in boxes:
                    if bx < ax1 and bx + bw > ax0 and by < ay1 and by + bh > ay0:
                        ax0, ay0 = min(ax0, bx - gap), min(ay0, by - gap)
                        ax1, ay1 = max(ax1, bx + bw + gap), max(ay1, by + bh + gap)
                        merged = changed = True
                    else:
                        rest.append((bx, by, bw, bh))
                boxes = rest
            out.append((ax0 + gap, ay0 + gap, ax1 - ax0 - 2 * gap, ay1 - ay0 - 2 * gap))
        boxes = out
    return boxes


def find_red_boxes(hsv):
    """Return merged red NG boxes [(x, y, w, h), ...] in RESULT pixels."""
    closed = cv2.morphologyEx(red_mask(hsv), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_KERNEL, CLOSE_KERNEL)))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= MIN_BOX_AREA and w >= MIN_BOX_W and h >= MIN_BOX_H:
            boxes.append((x, y, w, h))
    return _merge_boxes(boxes)


def cable_mask(hsv):
    """Filled+dilated mask of the large gold flex cable, to exclude its orange-hued edges."""
    m = cv2.morphologyEx(cv2.inRange(hsv, np.array(CABLE_LOWER), np.array(CABLE_UPPER)),
                         cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    filled = np.zeros_like(m)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) > CABLE_MIN_AREA:
            cv2.drawContours(filled, [c], -1, 255, -1)
    return cv2.dilate(filled, np.ones((CABLE_DILATE, CABLE_DILATE), np.uint8))


def _orange_boxes(orange_sub):
    """Defect-box candidates in an orange sub-mask, as local (x, y, w, h): close the
    orange into blobs, take each contour's bounding box, keep those big enough."""
    closed = cv2.morphologyEx(orange_sub, cv2.MORPH_CLOSE,
                              np.ones((DEFECT_CLOSE, DEFECT_CLOSE), np.uint8))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= DEFECT_MIN_AREA and w >= DEFECT_MIN_WH and h >= DEFECT_MIN_WH:
            out.append((x, y, w, h))
    return out


def find_defect_boxes(result_img):
    """Defect boxes per NG region. Returns [(x, y, w, h, kind), ...] in RESULT px.

    For each red NG box, recover the orange defect box(es) nested inside it (cable
    excluded), capturing ALL of them -- a zone can hold several non-overlapping defects.
    To keep the red outline's orange bleed from being mistaken for a box (or merging into
    a real one and ballooning it to the whole zone), the search runs on the zone with a
    PERIM_FRAME-px border blanked.

    A border sliver (thin, edge-hugging; see _is_border_sliver) is kept only when a real
    (non-sliver) box also anchors the zone -- then it is plausibly a second thin defect the
    machine boxed (e.g. ZCH31KWB). A sliver ALONE in a zone is the red-edge bleed, so we
    discard it and retry on the unframed zone (rescuing a real box that fills a thin zone
    and so touches the border); if nothing real turns up, fall back to the whole red zone.

    kind == 'tight'  -> recovered orange defect box
    kind == 'zone'   -> fallback: no orange box found, use the whole red box
    """
    hsv = cv2.cvtColor(result_img, cv2.COLOR_BGR2HSV)
    red_boxes = find_red_boxes(hsv)
    orange = cv2.bitwise_and(
        cv2.inRange(hsv, np.array(ORANGE_LOWER), np.array(ORANGE_UPPER)),
        cv2.bitwise_not(cable_mask(hsv)),
    )

    def real(boxes):                                      # a non-sliver box anchors the zone
        return any(not _is_border_sliver(x, y, w, h, rw, rh) for (x, y, w, h) in boxes)

    out = []
    for (rx, ry, rw, rh) in red_boxes:
        sub = orange[ry:ry + rh, rx:rx + rw]
        framed = sub.copy()                               # peel the red-outline bleed off the border
        framed[:PERIM_FRAME, :] = 0; framed[-PERIM_FRAME:, :] = 0
        framed[:, :PERIM_FRAME] = 0; framed[:, -PERIM_FRAME:] = 0
        cands = _orange_boxes(framed)
        if real(cands):
            chosen = cands                                # keep all (real box + any thin peers)
        else:                                             # only slivers (or nothing): try the whole zone
            chosen = [b for b in _orange_boxes(sub) if not _is_border_sliver(*b, rw, rh)]
        if chosen:
            out.extend((rx + x, ry + y, w, h, "tight") for (x, y, w, h) in chosen)
        else:
            out.append((rx, ry, rw, rh, "zone"))          # fallback to the whole NG zone
    return out


def discover_fail_results():
    recs = []
    for result_path in sorted(glob.glob(os.path.join(RESULT_IMAGE_DIR, f"*{IMAGE_EXT}"))):
        stem = os.path.splitext(os.path.basename(result_path))[0]
        parts = stem.split("_")
        if parts[-1] != FAIL_STATUS:
            continue
        base = "_".join(parts[:-1])
        raw_path = os.path.join(RAW_IMAGE_DIR, base + IMAGE_EXT)
        if os.path.exists(raw_path):
            date = parts[1][:8]
            zone = parts[2]
            view = parts[3]
            recs.append((parts[0], date, zone, view, raw_path, result_path))
    return recs


# Diag overlay colors (BGR) - deliberately NOT green/red/orange, which the machine itself uses.
DIAG_TIGHT_COL = (255, 255, 0)     # cyan  = recovered tight defect box (what gets cropped)
DIAG_ZONE_COL  = (255, 0, 255)     # magenta = whole NG zone (fallback crop)
_LEGEND_LINES = [
    "cyan/magenta = box THIS script detects & crops",
    "background:  green = OK zones   red = NG zone   orange = machine defect box",
    "gold diagonal = flex cable (not a box)",
]


def _draw_diag_legend(img):
    """Draw a semi-transparent explanatory header banner across the top of a diag image."""
    font, scale, thick, lh = cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2, 46
    banner_h = 24 + lh * len(_LEGEND_LINES)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, text in enumerate(_LEGEND_LINES):
        cv2.putText(img, text, (16, 20 + lh * (i + 1) - 12), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return banner_h


def main():
    for d in (OUT_SIDE_DIR, OUT_DIAG_DIR):
        os.makedirs(d, exist_ok=True)

    for serial, date, zone, view, raw_path, result_path in discover_fail_results():
        result_img = cv2.imread(result_path)
        raw_img = cv2.imread(raw_path)
        if result_img is None or raw_img is None:
            print(f"[{serial}] could not read images, skipping")
            continue

        res_h, res_w = result_img.shape[:2]
        raw_h, raw_w = raw_img.shape[:2]
        sx, sy = raw_w / res_w, raw_h / res_h          # RESULT -> RAW scale (== 2.0 here)

        boxes = find_defect_boxes(result_img)
        kinds = ", ".join(b[4] for b in boxes) or "none"
        print(f"[{serial}] {len(boxes)} defect box(es): {kinds}  (raw/result scale {sx:.2f}x{sy:.2f})")

        diag = result_img.copy()
        banner_h = _draw_diag_legend(diag)
        for (x, y, w, h, kind) in boxes:
            col = DIAG_TIGHT_COL if kind == "tight" else DIAG_ZONE_COL
            label = "TIGHT DEFECT (cropped)" if kind == "tight" else "NG ZONE (fallback crop)"
            cv2.rectangle(diag, (x, y), (x + w, y + h), col, 4)
            # keep the label clear of the top banner: below the box if it sits under the banner
            ly = y + h + 34 if y < banner_h + 24 else y - 12
            cv2.putText(diag, label, (x, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT_DIAG_DIR, f"{serial}_detection.png"),
                    cv2.resize(diag, None, fx=0.5, fy=0.5))

        for i, (x, y, w, h, kind) in enumerate(boxes):
            # fixed CROP_SIZE × CROP_SIZE window in RAW pixels, centred on defect centroid
            rx, ry, rw, rh = x * sx, y * sy, w * sx, h * sy
            half = CROP_SIZE / 2
            cx0 = int(rx + rw / 2 - half)
            cy0 = int(ry + rh / 2 - half)
            cx1 = cx0 + CROP_SIZE
            cy1 = cy0 + CROP_SIZE
            # shift inward if the window exceeds raw image bounds (never shrink the window)
            if cx0 < 0:
                cx1 -= cx0; cx0 = 0
            if cy0 < 0:
                cy1 -= cy0; cy0 = 0
            if cx1 > raw_w:
                cx0 -= cx1 - raw_w; cx1 = raw_w
            if cy1 > raw_h:
                cy0 -= cy1 - raw_h; cy1 = raw_h
            # clamp in case the raw image is smaller than CROP_SIZE
            cx0 = max(0, cx0); cy0 = max(0, cy0)
            crop = raw_img[cy0:cy1, cx0:cx1]
            if crop.size == 0:
                print(f"  {i}: empty crop, skipping")
                continue

            fname = f"{serial}_{date}_{zone}-{view}_{i}"
            crop_dir = os.path.join(OUT_CROP_BASE, f"{zone}-{view}")
            os.makedirs(crop_dir, exist_ok=True)
            cv2.imwrite(os.path.join(crop_dir, f"{fname}.png"), crop)

            # side-by-side: raw (downscaled) with BLUE crop border (+ defect box) | the crop
            box_col = (0, 140, 255) if kind == "tight" else (0, 0, 255)
            disp = raw_img.copy()
            cv2.rectangle(disp, (cx0, cy0), (cx1, cy1), (255, 0, 0), 10)            # BLUE crop border
            cv2.rectangle(disp, (int(rx), int(ry)),
                          (int(rx + rw), int(ry + rh)), box_col, 6)                 # defect / zone box
            disp_small = cv2.resize(disp, None, fx=0.28, fy=0.28)

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].imshow(cv2.cvtColor(disp_small, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"{serial}  RAW  (blue = crop border, {'orange = defect box' if kind == 'tight' else 'red = NG zone fallback'})")
            axes[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"crop {i} [{kind}]   {crop.shape[1]}x{crop.shape[0]} px")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            fig.savefig(os.path.join(OUT_SIDE_DIR, f"{fname}.png"), dpi=90)
            plt.close(fig)
            print(f"  {i} [{kind}]: result=({x},{y},{w},{h}) -> fixed crop raw=({cx0},{cy0})-({cx1},{cy1})"
                  f"  {crop.shape[1]}x{crop.shape[0]} px")


if __name__ == "__main__":
    main()
