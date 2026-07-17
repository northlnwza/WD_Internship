"""
AVI scan -- single-file inspection app (self-contained, no local imports).

Two crop sources, chosen by a Mode toggle in the Scan tab:

  Result-mapping : the classic rejudge flow. Watch a Fail folder of RESULT
                   images (machine bounding boxes), find each defect box, map its
                   centre RESULT->RAW, cut a CROP_SIZE window from the RAW and
                   predict + judge. One Fail image = one unit.
  Zoning         : watch a RAW folder directly. Crop each RAW with a PRODUCT's
                   user-drawn ROI shapes (rect / circle / polygon), mask+clip to
                   the region so the model only sees inside it. One RAW = one unit
                   (holding one crop per zone). The RAW basename is the identity.

A second tab (ROI Editor) draws and saves those product zones to rois.json.

Everything (box recovery, engine, judging, output writer, review GUI, zoning
geometry, ROI store, ROI editor) is copied inline -- the only imports are
third-party libraries.

Usage:  python avi_scan.py
"""

import csv
import glob
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


# =============================================================== box recovery ==
# (copied from crop_ng_regions.py: find_defect_boxes chain + constants)

# ---- Config -----------------------------------------------------------------
RAW_IMAGE_DIR    = "./images/raw"
RESULT_IMAGE_DIR = "./images/result"
OUT_CROP_DIR     = "./crop"
TEST_MODE        = False           # True → saves into crop/{station}/test/
OUT_SIDE_DIR     = "./debug_out/sidebyside"
OUT_DIAG_DIR     = "./debug_out/diag"
IMAGE_EXT        = ".jpg"
FAIL_STATUS      = "Fail"

CROP_SIZE        = 640          # fixed crop window (RAW pixels); centred on defect centroid
TILE_OVERLAP     = 128          # zoning tiles: px shared by neighbouring tiles -- wider than
                                # any single defect, so each defect fits whole in >=1 tile
SIDE_SCALE       = 0.28         # downscale of the RAW overview panel in the side-by-side figure
NUM_WORKERS      = os.cpu_count() or 1   # units processed in parallel (each unit is independent)

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



# ============================================ core pipeline / engine / GUI shell ==
# (copied from rejudge_gui.py: app_dir .. App)

def app_dir():
    """Folder the app lives in: next to the .exe when frozen (PyInstaller),
    else next to this file. config.yaml, the weights dirs and rejudge_output
    are all resolved from here."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


CONFIG_PATH = os.path.join(app_dir(), "config.yaml")

# Verdicts (CONTEXT.md): worst-crop ladder NG > Review > Cleaning > Overkill.
NG, REVIEW, CLEANING, OVERKILL = "NG", "Review", "Cleaning", "Overkill"
LADDER = {NG: 3, REVIEW: 2, CLEANING: 1, OVERKILL: 0}
VERDICTS = [NG, OVERKILL, CLEANING, REVIEW]          # button/key order 1..4

VERDICT_COL = {NG: (0, 0, 255), REVIEW: (255, 0, 255),
               CLEANING: (0, 215, 255), OVERKILL: (0, 200, 0)}   # BGR

TRUNC_TOL = 3          # px: mask this close to the crop border counts as touching
GROW_STEPS = 3         # window sizes tried per truncated mask: 1x, 2x, 4x CROP_SIZE
MERGE_OVERLAP = 0.3    # masks sharing this fraction of the smaller mask are one defect
OVERLAY_MAX_W = 1600   # stored overlay width cap (display + saved copy)
RAW_WAIT_S = 30        # s: how long a new Fail image may wait for its RAW to arrive


# ---------------------------------------------------------------- settings --

@dataclass
class Settings:
    """rejudge: section of config.yaml. None = not decided yet -> the classes
    that need the value fall back to Review instead of guessing."""
    conf_threshold: float = 0.25
    near_limit_margin: float = 0.80
    poll_seconds: float = 2.0
    scratch_length_mm: float | None = None
    scuffmark_area_mm2: float | None = None
    max_defect_count: int | None = None
    max_total_area_mm2: float | None = None
    default_mm_per_px: float | None = None
    per_station_view: dict = field(default_factory=dict)
    weights_dirs: list = field(default_factory=lambda: ["models", "train history"])
    output_dir: str = "rejudge_output"

    @staticmethod
    def load(path=CONFIG_PATH):
        s = Settings()
        try:
            with open(path, encoding="utf-8") as f:
                rj = (yaml.safe_load(f) or {}).get("rejudge") or {}
        except OSError:
            return s
        s.conf_threshold = rj.get("conf_threshold", s.conf_threshold)
        s.near_limit_margin = rj.get("near_limit_margin", s.near_limit_margin)
        s.poll_seconds = rj.get("poll_seconds", s.poll_seconds)
        crit = rj.get("criteria") or {}
        s.scratch_length_mm = crit.get("scratch_length_mm")
        s.scuffmark_area_mm2 = crit.get("scuffmark_area_mm2")
        lim = rj.get("unit_limits") or {}
        s.max_defect_count = lim.get("max_defect_count")
        s.max_total_area_mm2 = lim.get("max_total_area_mm2")
        cal = rj.get("calibration") or {}
        s.default_mm_per_px = cal.get("default_mm_per_px")
        s.per_station_view = dict(cal.get("per_station_view") or {})
        s.weights_dirs = rj.get("weights_dirs", s.weights_dirs)
        s.output_dir = rj.get("output_dir", s.output_dir)
        return s

    def mm_per_px(self, station_view):
        return self.per_station_view.get(station_view, self.default_mm_per_px)


# ------------------------------------------------------------------- model --

@dataclass
class Detection:
    class_name: str
    conf: float
    poly: np.ndarray            # Nx2 float, RAW-image pixel coords
    length_px: float = 0.0      # longest min-area-rect side
    area_px: float = 0.0        # mask area
    truncated: bool = False     # still touching the border after max grow
    # judged fields (recomputed when settings change):
    verdict: str = REVIEW
    measure_text: str = ""      # e.g. "L=2.31mm" / "A=140px²"
    near_limit: bool = False
    note: str = ""

    def bbox(self):
        x0, y0 = self.poly.min(axis=0)
        x1, y1 = self.poly.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)


@dataclass
class CropResult:
    index: int
    kind: str                   # 'tight' | 'zone' (crop_ng_regions vocabulary)
    window: tuple               # (x0, y0, x1, y1) in RAW px
    detections: list = field(default_factory=list)
    verdict: str = REVIEW
    note: str = ""
    shape: dict | None = None   # ROI shape dict (rect/circle/poly); None => draw window rect


@dataclass
class UnitResult:
    serial: str
    date: str
    station: str
    view: str
    raw_path: str
    result_path: str
    crops: list = field(default_factory=list)
    defect_count: int = 0
    total_area_px: float = 0.0
    total_area_mm2: float | None = None
    suggested: str = REVIEW
    unit_note: str = ""
    human: str = ""
    human_time: str = ""
    error: str = ""
    overlay_base: np.ndarray | None = None  # downscaled clean RAW (annotations drawn on demand)
    overlay_scale: float = 1.0              # RAW px -> overlay_base px
    crop_bases: list = field(default_factory=list)   # clean full-res crops (BGR)

    @property
    def station_view(self):
        return f"{self.station}-{self.view}"


# ------------------------------------------------------- geometry helpers --

def crop_window(raw_w, raw_h, cx, cy, size):
    """size x size window centred on (cx, cy), shifted inward at RAW borders
    (never shrunk unless the RAW itself is smaller than size)."""
    half = size / 2
    x0, y0 = int(cx - half), int(cy - half)
    x1, y1 = x0 + size, y0 + size
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > raw_w:
        x0 -= x1 - raw_w; x1 = raw_w
    if y1 > raw_h:
        y0 -= y1 - raw_h; y1 = raw_h
    return max(0, x0), max(0, y0), x1, y1


def touches_border(poly, window, raw_w, raw_h, tol=TRUNC_TOL):
    """True if a RAW-coord polygon hugs a window edge that is NOT also the RAW
    image edge (at the RAW edge the defect genuinely ends -- nothing to grow)."""
    x0, y0, x1, y1 = window
    px0, py0 = poly.min(axis=0)
    px1, py1 = poly.max(axis=0)
    return ((px0 - x0 <= tol and x0 > 0) or (py0 - y0 <= tol and y0 > 0)
            or (x1 - px1 <= tol and x1 < raw_w) or (y1 - py1 <= tol and y1 < raw_h))


def measure_poly(poly):
    """(longest min-area-rect side, area) of a polygon, in px / px²."""
    pts = poly.astype(np.float32)
    if len(pts) < 3:
        return 0.0, 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return float(max(w, h)), float(cv2.contourArea(pts))


def bbox_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def _rasterize(polys, x0, y0, w, h):
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [(p - [x0, y0]).astype(np.int32)], 1)
    return m


def mask_overlap_frac(poly_a, poly_b):
    """Shared pixels as a fraction of the SMALLER mask, so containment and
    partial overlap both score high (bbox IoU does not, for thin diagonal
    masks). 0.0 when the bounding boxes do not even intersect."""
    ax0, ay0 = poly_a.min(axis=0)
    ax1, ay1 = poly_a.max(axis=0)
    bx0, by0 = poly_b.min(axis=0)
    bx1, by1 = poly_b.max(axis=0)
    if ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0:
        return 0.0
    x0, y0 = int(min(ax0, bx0)), int(min(ay0, by0))
    w = int(np.ceil(max(ax1, bx1))) - x0 + 1
    h = int(np.ceil(max(ay1, by1))) - y0 + 1
    a = _rasterize([poly_a], x0, y0, w, h)
    b = _rasterize([poly_b], x0, y0, w, h)
    inter = int(np.logical_and(a, b).sum())
    smaller = min(int(a.sum()), int(b.sum()))
    return inter / smaller if smaller else 0.0


def union_area_px(polys):
    """Union area (px²) of polygons via rasterization -- overlapping regions
    are counted once. Boundary-inclusive, so slightly above cv2.contourArea
    for a single polygon; per-detection gating keeps using measure_poly."""
    if not polys:
        return 0.0
    pts = np.vstack(polys)
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    m = _rasterize(polys, int(x0), int(y0), int(x1 - x0) + 1, int(y1 - y0) + 1)
    return float(m.sum())


# ------------------------------------------------------------ verdict rules --

def judge_detection(det: Detection, mm_per_px, s: Settings):
    """Fill det.verdict / measure_text / near_limit / note from the stored px
    measurements (docs/adr/0001). Pure re-mapping -- no prediction involved."""
    name = det.class_name.lower()
    det.near_limit = False

    if det.truncated:
        det.verdict, det.note = REVIEW, "mask still cut off at max window - size unreliable"
        det.measure_text = _fmt_len(det.length_px, mm_per_px)
        return

    if name.startswith("contam"):
        det.verdict, det.note = CLEANING, "contamination - requires cleaning"
        det.measure_text = _fmt_area(det.area_px, mm_per_px)
        return

    if name == "scratch":
        det.measure_text = _fmt_len(det.length_px, mm_per_px)
        _gate(det, det.length_px, mm_per_px, s.scratch_length_mm, s, linear=True)
        return

    if name == "scuffmark":
        det.measure_text = _fmt_area(det.area_px, mm_per_px)
        _gate(det, det.area_px, mm_per_px, s.scuffmark_area_mm2, s, linear=False)
        return

    # spot / other / any unknown class (e.g. legacy 3-class weights): human only
    det.verdict, det.note = REVIEW, f"'{det.class_name}' needs human judgment"
    det.measure_text = _fmt_area(det.area_px, mm_per_px)


def _gate(det, px_value, mm_per_px, limit, s: Settings, linear):
    if mm_per_px is None or limit is None:
        det.verdict = REVIEW
        det.note = "size criteria / calibration not set (config.yaml rejudge:)"
        return
    mm_value = px_value * (mm_per_px if linear else mm_per_px ** 2)
    if mm_value > limit:
        det.verdict, det.note = NG, f"over limit ({mm_value:.2f} > {limit:g})"
    elif mm_value >= s.near_limit_margin * limit:
        det.verdict, det.near_limit = OVERKILL, True
        det.note = f"under limit but close ({mm_value:.2f} vs {limit:g}) - check"
    else:
        det.verdict, det.note = OVERKILL, f"under limit ({mm_value:.2f} vs {limit:g})"


def _fmt_len(px, mm_per_px):
    return f"L={px * mm_per_px:.2f}mm" if mm_per_px else f"L={px:.0f}px"


def _fmt_area(px2, mm_per_px):
    return f"A={px2 * mm_per_px ** 2:.2f}mm²" if mm_per_px else f"A={px2:.0f}px²"


def cluster_defects(dets):
    """Group detections into physical defects. Crop windows overlap and one
    mark can be predicted several times -- even as different classes -- so
    detections whose masks share > MERGE_OVERLAP of the smaller mask are one
    defect. Returns a list of clusters (each a list of Detections)."""
    parent = list(range(len(dets)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if find(i) != find(j) and \
                    mask_overlap_frac(dets[i].poly, dets[j].poly) > MERGE_OVERLAP:
                parent[find(j)] = find(i)
    groups = {}
    for i, d in enumerate(dets):
        groups.setdefault(find(i), []).append(d)
    return list(groups.values())


def judge_unit(unit: UnitResult, s: Settings):
    """Crop verdicts -> worst-crop ladder -> unit count/total-area limits.
    Recomputable from stored px measurements when settings change."""
    mm = s.mm_per_px(unit.station_view)
    for crop in unit.crops:
        for det in crop.detections:
            judge_detection(det, mm, s)
        if crop.detections:
            crop.verdict = max((d.verdict for d in crop.detections), key=LADDER.get)
            crop.note = ""
        else:
            crop.verdict = REVIEW
            crop.note = "model found nothing - machine/model disagree"

    # Unit limits count physical marks: overlapping predictions collapse to one
    # defect and shared area counts once. Contamination is dispositioned by
    # Cleaning, so it does not gate the unit NG via count/area.
    gated = [d for c in unit.crops for d in c.detections
             if not d.class_name.lower().startswith("contam")]
    unit.defect_count = len(cluster_defects(gated))
    unit.total_area_px = union_area_px([d.poly for d in gated])
    unit.total_area_mm2 = unit.total_area_px * mm ** 2 if mm else None

    unit.suggested = max((c.verdict for c in unit.crops), key=LADDER.get) \
        if unit.crops else REVIEW
    unit.unit_note = ""
    if s.max_defect_count is not None and unit.defect_count > s.max_defect_count:
        unit.suggested = NG
        unit.unit_note = f"defect count {unit.defect_count} > limit {s.max_defect_count}"
    if (s.max_total_area_mm2 is not None and unit.total_area_mm2 is not None
            and unit.total_area_mm2 > s.max_total_area_mm2):
        unit.suggested = NG
        unit.unit_note = (unit.unit_note + "; " if unit.unit_note else "") + \
            f"total area {unit.total_area_mm2:.2f}mm² > limit {s.max_total_area_mm2:g}"


# ----------------------------------------------------------------- engine --

def parse_fail_name(path):
    """(serial, date, station, view) from a Fail-image path. Canonical layout
    {serial}_{datetime}_{station}_{view}[_..._Fail]; whatever is missing falls
    back to '?' (station/view) or the file's modification day (date), so any
    *.jpg dropped in the Fail folder can still be processed and filed."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    if parts[-1] == FAIL_STATUS:
        parts = parts[:-1]
    if len(parts) > 1 and len(parts[1]) >= 8 and parts[1][:8].isdigit():
        date = parts[1][:8]
    else:
        try:
            date = time.strftime("%Y%m%d", time.localtime(os.path.getmtime(path)))
        except OSError:
            date = time.strftime("%Y%m%d")
    station = parts[2] if len(parts) > 2 else "?"
    view = parts[3] if len(parts) > 3 else "?"
    return parts[0], date, station, view


def _dt_key(parts, path):
    """Sortable YYYYmmddHHMMSS string for an image: the filename's datetime
    token when present, else the file's modification time."""
    if len(parts) > 1 and parts[1].isdigit() and len(parts[1]) >= 8:
        return parts[1][:14].ljust(14, "0")
    try:
        return time.strftime("%Y%m%d%H%M%S", time.localtime(os.path.getmtime(path)))
    except OSError:
        return "99999999999999"


def find_raw_for(serial, station, view, raw_dir, fail_stem=None):
    """RAW image for a Fail image, matched by serial (the first _-token).
    Pool preference: exact stem twin (Fail name minus _Fail) > same
    station+view > any serial match. Within the pool: the RAW taken closest
    AT-OR-BEFORE the Fail image's time -- a Fail can only come from an earlier
    exposure, and a later retest RAW may show the unit repositioned, so it
    must never be measured against this Fail's boxes. Only when nothing
    at-or-before exists (clock skew) is the closest later RAW used. None when
    the serial has no RAW yet -- it may still be copying; the watcher retries."""
    fail_dt = None
    if fail_stem:
        parts = fail_stem.split("_")
        if parts[-1] == FAIL_STATUS:
            twin = os.path.join(raw_dir, "_".join(parts[:-1]) + IMAGE_EXT)
            if os.path.exists(twin):
                return twin
        if len(parts) > 1 and parts[1].isdigit() and len(parts[1]) >= 8:
            fail_dt = parts[1][:14].ljust(14, "0")
    cands = []
    for p in (glob.glob(os.path.join(raw_dir, f"{serial}_*{IMAGE_EXT}"))
              + glob.glob(os.path.join(raw_dir, f"{serial}{IMAGE_EXT}"))):
        parts = os.path.splitext(os.path.basename(p))[0].split("_")
        if parts[-1] != FAIL_STATUS:            # a stray RESULT copy is not a RAW
            cands.append((_dt_key(parts, p), parts, p))
    same_sv = [c for c in cands if c[1][2:4] == [station, view]]
    pool = same_sv or cands
    if not pool:
        return None
    if fail_dt:
        before = [c for c in pool if c[0] <= fail_dt]
        return max(before)[2] if before else min(pool)[2]
    return max(pool)[2]                         # no Fail timestamp: newest known


PIC_LEAF = "Activescale"     # the only image subfolder under a {product}/{day} day folder


def _pic_subdirs(path):
    """Immediate subfolder names of `path` (empty set if it is not a dir)."""
    try:
        return {n for n in os.listdir(path) if os.path.isdir(os.path.join(path, n))}
    except OSError:
        return set()


def pic_products(root, both_sides):
    """Product folder names under the picture-tree `root`. both_sides=True
    (Result-mapping) -> present under BOTH Fail/ and Image/; False (Zoning) ->
    Image/ only. Sorted."""
    img = _pic_subdirs(os.path.join(root, "Image"))
    if both_sides:
        img &= _pic_subdirs(os.path.join(root, "Fail"))
    return sorted(img)


def pic_leaf(root, side, product, day):
    """The Activescale image folder for one side/product/day."""
    return os.path.join(root, side, product, day, PIC_LEAF)


def product_from_raw_path(path):
    """Product folder name for a RAW that lives under .../Image/{product}/{day}/
    Activescale/file. '' when the path is not in that tree layout -- so the ROI
    Editor labels zones with the SAME identity the scan uses."""
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 4 and parts[-2] == PIC_LEAF:
        return parts[-4]                         # .../product/day/Activescale/file
    return ""


def pic_days(root, product, both_sides):
    """Day folders for `product` that carry an Activescale leaf on the needed
    side(s): Image always, plus Fail when both_sides. Newest first."""
    days = {d for d in _pic_subdirs(os.path.join(root, "Image", product))
            if os.path.isdir(pic_leaf(root, "Image", product, d))
            and (not both_sides or os.path.isdir(pic_leaf(root, "Fail", product, d)))}
    return sorted(days, reverse=True)


def find_weights(dirs, root):
    """All *.pt under the configured dirs, labeled 'run-folder/file.pt'."""
    out = {}
    for d in dirs:
        base = os.path.join(root, d)
        for p in sorted(glob.glob(os.path.join(base, "**", "*.pt"), recursive=True)):
            rel = os.path.relpath(p, root)
            out[rel] = p
    return out


class Engine:
    def __init__(self, weights_path, settings: Settings):
        self.settings = settings
        self.weights_path = weights_path
        self.model = None
        self.device = "cpu"       # resolved in load(): 0 (CUDA) when a GPU is present
        self.half = False         # FP16 only on GPU
        self.batch = 8            # zones per predict call (raised on GPU)

    def load(self):
        import torch
        from ultralytics import YOLO
        # AVI_FORCE_CPU=1 ignores the GPU (diagnosis knob: isolates GPU-path
        # failures from CPU-path ones on the server).
        force_cpu = os.environ.get("AVI_FORCE_CPU", "").strip() not in ("", "0")
        print("[engine] probing CUDA...", flush=True)  # hang right here = driver-level cuInit hang
        if not force_cpu and torch.cuda.is_available():   # auto-detect the server GPU
            self.device, self.half, self.batch = 0, True, 32
            cap = torch.cuda.get_device_capability(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            print(f"[engine] gpu={torch.cuda.get_device_name(0)} "
                  f"capability={cap[0]}.{cap[1]} "
                  f"vram={free_b / 2**30:.1f}GiB free / {total_b / 2**30:.1f}GiB "
                  f"wheel_archs={torch.cuda.get_arch_list()}", flush=True)
            # Pre-Volta GPUs (capability < 7.0, e.g. Pascal Quadro/GTX 10xx) run
            # FP16 up to 64x slower than FP32: a "frozen" scan that is really one
            # many-minute batch. FP32 loses nothing in accuracy, so force it there.
            if cap < (7, 0):
                self.half = False
            # Server-side bisect knobs (run_avi_scan_fp32.bat / run_avi_scan_batch4.bat).
            if os.environ.get("AVI_NO_HALF", "").strip() not in ("", "0"):
                self.half = False
            self.batch = max(1, int(os.environ.get("AVI_GPU_BATCH", self.batch)))
        else:                                  # CPU box: use every core
            self.device, self.half, self.batch = "cpu", False, 8
            # AVI_CPU_THREADS overrides the all-cores default (diagnosis knob:
            # many-core servers can thrash on full-width intra-op threading).
            threads = int(os.environ.get("AVI_CPU_THREADS", os.cpu_count() or 1))
            torch.set_num_threads(max(1, threads))
        print(f"[engine] torch={torch.__version__} cuda={torch.cuda.is_available()} "
              f"device={self.device} half={self.half} batch={self.batch} "
              f"threads={torch.get_num_threads()} cores={os.cpu_count()}", flush=True)
        t0 = time.time()
        self.model = YOLO(self.weights_path)
        print(f"[engine] weights loaded in {time.time() - t0:.1f}s", flush=True)
        if self.device != "cpu":
            # The first CUDA inference pays context init + cuDNN autotune + (if
            # the wheel ships no SASS for this GPU) a PTX JIT recompile of every
            # kernel -- worst case many minutes at 100% CPU. Paying it here pins
            # any such stall to this breadcrumb instead of the first real unit.
            t0 = time.time()
            self.model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640,
                               verbose=False, device=self.device, half=self.half)
            print(f"[engine] gpu warmup predict in {time.time() - t0:.1f}s", flush=True)

    def _decode(self, r):
        """[(class_name, conf, poly_local Nx2)] from one ultralytics Result."""
        out = []
        if r.masks is None:
            return out
        for poly, cls, conf in zip(r.masks.xy, r.boxes.cls, r.boxes.conf):
            if len(poly) >= 3:
                out.append((str(self.model.names[int(cls)]), float(conf),
                            np.asarray(poly, np.float32)))
        return out

    def predict(self, img):
        """[(class_name, conf, poly_local Nx2)] for one BGR crop."""
        return self._decode(self.model.predict(
            img, conf=self.settings.conf_threshold, imgsz=640, verbose=False,
            device=self.device, half=self.half)[0])

    def predict_many(self, imgs):
        """Detections for a list of BGR crops -> a list of per-crop detection
        lists, order preserved.

        On GPU we batch (one model call per `batch`-sized chunk) -- the parallel
        hardware makes a batch far cheaper than N separate calls. On CPU we do NOT
        batch: measured, a CPU batch is slower than sequential single predicts (no
        parallelism, worse cache behaviour) and it perturbs results via float
        jitter -- so CPU runs the exact per-zone path, identical to before."""
        if self.device == "cpu":
            return [self.predict(im) for im in imgs]
        out = []
        n_chunks = (len(imgs) + self.batch - 1) // self.batch
        for i in range(0, len(imgs), self.batch):
            t0 = time.time()
            results = self.model.predict(
                imgs[i:i + self.batch], conf=self.settings.conf_threshold,
                imgsz=640, verbose=False, device=self.device, half=self.half)
            print(f"[engine] gpu chunk {i // self.batch + 1}/{n_chunks} "
                  f"({len(results)} zones) in {time.time() - t0:.1f}s", flush=True)
            out.extend(self._decode(r) for r in results)
        return out

    def _refine(self, raw, det: Detection):
        """Truncated-mask rule: re-crop centred on the mask and re-predict,
        growing the window (1x, 2x, 4x) until the mask clears the border.
        If it never does, det.truncated stays True (size unreliable -> Review)."""
        raw_h, raw_w = raw.shape[:2]
        # Only reached because the mask touched its prediction window's border;
        # cleared solely by a re-prediction that clears a real window's border.
        det.truncated = True
        for step in range(GROW_STEPS):
            size = CROP_SIZE * (2 ** step)
            cx, cy = det.poly.mean(axis=0)
            win = crop_window(raw_w, raw_h, cx, cy, size)
            x0, y0, x1, y1 = win
            match = None
            for name, conf, local in self.predict(raw[y0:y1, x0:x1]):
                if name != det.class_name:
                    continue
                cand = local + np.array([x0, y0], np.float32)
                iou = bbox_iou(det.bbox(), Detection(name, conf, cand).bbox())
                if match is None or iou > match[0]:
                    match = (iou, conf, cand)
            if match and match[0] > 0:
                det.conf, det.poly = match[1], match[2]
                if not touches_border(det.poly, win, raw_w, raw_h):
                    det.truncated = False
                    return
            # still cut off (or not re-found) at this size: try larger

    def process_unit(self, rec):
        serial, date, station, view, raw_path, result_path = rec
        unit = UnitResult(serial, date, station, view, raw_path, result_path)
        result_img = cv2.imread(result_path)
        raw = cv2.imread(raw_path)
        if result_img is None or raw is None:
            unit.error = "could not read RAW/RESULT image"
            unit.suggested = REVIEW
            return unit

        raw_h, raw_w = raw.shape[:2]
        sx, sy = raw_w / result_img.shape[1], raw_h / result_img.shape[0]

        for i, (x, y, w, h, kind) in enumerate(find_defect_boxes(result_img)):
            cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
            win = crop_window(raw_w, raw_h, cx, cy, CROP_SIZE)
            x0, y0, x1, y1 = win
            crop = CropResult(i, kind, win)
            for name, conf, local in self.predict(raw[y0:y1, x0:x1]):
                det = Detection(name, conf, local + np.array([x0, y0], np.float32))
                if touches_border(det.poly, win, raw_w, raw_h):
                    self._refine(raw, det)
                det.length_px, det.area_px = measure_poly(det.poly)
                crop.detections.append(det)
            unit.crops.append(crop)

        judge_unit(unit, self.settings)
        unit.overlay_scale = min(1.0, OVERLAY_MAX_W / raw_w)
        unit.overlay_base = cv2.resize(raw, None, fx=unit.overlay_scale, fy=unit.overlay_scale)
        unit.crop_bases = [raw[c.window[1]:c.window[3], c.window[0]:c.window[2]].copy()
                           for c in unit.crops]
        return unit


# -------------------------------------------------------------- rendering --

def _label(img, text, x, y, color):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)


def _draw_roi_outline(img, shape, window, s, colour, thick, offset=(0, 0)):
    """Outline an ROI in its true form. `shape` (rect/circle/poly, RAW px) wins;
    if None, fall back to the rectangular `window`. Coords are shifted by `offset`
    then scaled by `s` (overlay: offset 0, s=scale; crop: offset=origin, s=1)."""
    ox, oy = offset
    t = shape.get("type") if shape else None
    if t == "circle":
        cv2.circle(img, (int((shape["cx"] - ox) * s), int((shape["cy"] - oy) * s)),
                   max(1, int(shape["r"] * s)), colour, thick)
    elif t == "poly":
        pts = np.array([[int((x - ox) * s), int((y - oy) * s)] for x, y in shape["pts"]],
                       np.int32)
        cv2.polylines(img, [pts], True, colour, thick)
    else:
        x0, y0, x1, y1 = window
        cv2.rectangle(img, (int((x0 - ox) * s), int((y0 - oy) * s)),
                      (int((x1 - ox) * s), int((y1 - oy) * s)), colour, thick)


def render_overlay(unit: UnitResult, annotate=True):
    """Downscaled RAW; with annotate, crop windows (blue) and per-detection
    masks colored by verdict -- where on the unit each defect sits."""
    img = unit.overlay_base.copy()
    if not annotate:
        return img
    s = unit.overlay_scale
    for crop in unit.crops:
        x0, y0, x1, y1 = [int(v * s) for v in crop.window]
        _draw_roi_outline(img, crop.shape, crop.window, s, (255, 0, 0), 2)
        _label(img, f"crop {crop.index} [{crop.verdict}]", x0 + 4, max(16, y0 - 6), (255, 200, 0))
        for det in crop.detections:
            col = VERDICT_COL[det.verdict]
            pts = (det.poly * s).astype(np.int32)
            cv2.polylines(img, [pts], True, col, 2)
            bx, by = pts.min(axis=0)
            _label(img, f"{det.class_name} {det.conf:.2f} {det.measure_text}",
                   bx, max(14, by - 4), col)
    return img


def render_crop(unit: UnitResult, idx, annotate=True):
    """Full-res crop; with annotate, masks + class + conf + measurement + verdict."""
    crop: CropResult = unit.crops[idx]
    img = unit.crop_bases[idx].copy()
    if not annotate:
        return img
    x0, y0 = crop.window[:2]
    if crop.shape and crop.shape.get("type") != "rect":
        _draw_roi_outline(img, crop.shape, crop.window, 1.0, (255, 0, 0), 1, offset=(x0, y0))
    for det in crop.detections:
        col = VERDICT_COL[det.verdict]
        pts = (det.poly - [x0, y0]).astype(np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], col)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
        cv2.polylines(img, [pts], True, col, 2)
        bx, by = pts.min(axis=0)
        flag = " !NEAR-LIMIT" if det.near_limit else ""
        _label(img, f"{det.class_name} {det.conf:.2f} {det.measure_text} "
                    f"-> {det.verdict}{flag}", max(2, bx), max(16, by - 6), col)
    if not crop.detections:
        _label(img, "no detection -> Review", 8, 24, VERDICT_COL[REVIEW])
    return img


# ------------------------------------------------------------------ output --

class DateWriter:
    """rejudge_output/{YYYYMMDD}/{units.csv, crops.csv, images/...} -- outputs
    grouped by the date in the unit's filename, so a continuously watching
    session files each unit with its production day. Rewrites merge with what
    is already on disk: rows from other sessions/files are preserved and this
    session's units replace only their own rows (keyed by result_file, falling
    back to serial|station_view for old-format rows). units.csv records
    suggestion AND human decision side by side (adr/0002)."""

    UNIT_COLS = ["serial", "date", "station_view", "crops", "defect_count",
                 "total_area_mm2", "suggested", "unit_note", "human_verdict",
                 "human_time", "error", "result_file"]
    CROP_COLS = ["serial", "station_view", "crop", "kind", "class", "conf",
                 "measure", "verdict", "near_limit", "truncated", "note",
                 "result_file"]

    def __init__(self, out_root):
        self.root = out_root

    def _date_dir(self, date):
        d = os.path.join(self.root, date or "unknown-date")
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        return d

    def recorded(self, date):
        """result_file basenames already in this date's units.csv (earlier
        sessions) -- the watcher skips these on restart."""
        out = set()
        try:
            with open(os.path.join(self.root, date, "units.csv"),
                      newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    if row.get("result_file"):
                        out.add(row["result_file"])
        except OSError:
            pass
        return out

    def save_images(self, unit: UnitResult):
        d = os.path.join(self._date_dir(unit.date), "images",
                         f"{unit.serial}_{unit.station_view}")
        os.makedirs(d, exist_ok=True)
        if unit.overlay_base is not None:
            cv2.imwrite(os.path.join(d, "overlay.jpg"), render_overlay(unit))
        for i, crop in enumerate(unit.crops):
            cv2.imwrite(os.path.join(d, f"crop_{crop.index}_{crop.verdict}.png"),
                        render_crop(unit, i))

    SERIAL_COLS = ["serial", "date", "inspections", "judged",
                   "worst_suggested", "worst_human", "complete"]

    def write_all(self, units, only_date=None):
        by_date = {}
        for u in units:
            if only_date is None or u.date == only_date:
                by_date.setdefault(u.date, []).append(u)
        for date, us in by_date.items():
            d = self._date_dir(date)
            own = set()
            for u in us:
                own.add(os.path.basename(u.result_path))
                own.add(f"{u.serial}|{u.station_view}")
            merged = self._merge_write(os.path.join(d, "units.csv"), self.UNIT_COLS,
                                       [self._unit_row(u) for u in us], own)
            self._write_serials(d, date, merged)
            self._merge_write(os.path.join(d, "crops.csv"), self.CROP_COLS,
                              [r for u in us for r in self._crop_rows(u)], own)

    def _write_serials(self, ddir, date, unit_rows):
        """serials.csv: the unit disposition -- worst verdict per serial across
        its inspections of this date. Derived from the fully merged unit rows,
        so it always reflects every session's latest state."""
        by_serial = {}
        for r in unit_rows:
            by_serial.setdefault(r.get("serial") or "", []).append(r)

        def worst(verdicts):
            return max(verdicts, key=lambda v: LADDER.get(v, LADDER[REVIEW])) \
                if verdicts else ""

        rows = []
        for serial in sorted(by_serial):
            rs = by_serial[serial]
            humans = [r.get("human_verdict") or "" for r in rs if r.get("human_verdict")]
            rows.append({"serial": serial, "date": date, "inspections": len(rs),
                         "judged": len(humans),
                         "worst_suggested": worst([r.get("suggested") or "" for r in rs
                                                   if r.get("suggested")]),
                         "worst_human": worst(humans),
                         "complete": len(humans) == len(rs)})
        self._write_rows(os.path.join(ddir, "serials.csv"), self.SERIAL_COLS, rows)

    @staticmethod
    def _row_key(row):
        return row.get("result_file") or \
            f"{row.get('serial', '')}|{row.get('station_view', '')}"

    def _merge_write(self, path, cols, rows, own_keys):
        """Returns the full merged row set that was written."""
        kept = []
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                kept = [r for r in csv.DictReader(f)
                        if self._row_key(r) not in own_keys]
        except OSError:
            pass
        merged = [{c: (r.get(c) or "") for c in cols} for r in kept] + rows
        self._write_rows(path, cols, merged)
        return merged

    @staticmethod
    def _write_rows(path, cols, rows):
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)   # a locked target fails here with the original intact

    @staticmethod
    def _unit_row(u: UnitResult):
        return {"serial": u.serial, "date": u.date, "station_view": u.station_view,
                "crops": len(u.crops), "defect_count": u.defect_count,
                "total_area_mm2": "" if u.total_area_mm2 is None else f"{u.total_area_mm2:.3f}",
                "suggested": u.suggested, "unit_note": u.unit_note,
                "human_verdict": u.human, "human_time": u.human_time,
                "error": u.error, "result_file": os.path.basename(u.result_path)}

    @staticmethod
    def _crop_rows(u: UnitResult):
        rows, rf = [], os.path.basename(u.result_path)
        for c in u.crops:
            if not c.detections:
                rows.append({"serial": u.serial, "station_view": u.station_view,
                             "crop": c.index, "kind": c.kind, "class": "",
                             "conf": "", "measure": "", "verdict": c.verdict,
                             "near_limit": "", "truncated": "", "note": c.note,
                             "result_file": rf})
            for d in c.detections:
                rows.append({"serial": u.serial, "station_view": u.station_view,
                             "crop": c.index, "kind": c.kind, "class": d.class_name,
                             "conf": f"{d.conf:.3f}", "measure": d.measure_text,
                             "verdict": d.verdict, "near_limit": d.near_limit,
                             "truncated": d.truncated, "note": d.note,
                             "result_file": rf})
        return rows


# --------------------------------------------------------------------- GUI --

class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root = root
        root.title("AVI NG Re-judgment")
        root.geometry("1500x900")

        self.settings = Settings.load()
        self.project = app_dir()
        self.weights = find_weights(self.settings.weights_dirs, self.project)
        self.units: list[UnitResult] = []
        self.current = None          # index into self.units
        self.current_crop = 0
        self.writer = None
        self.q = queue.Queue(maxsize=8)   # backpressure: a GPU-speed worker that
                                          # outruns the GUI must idle, not balloon
                                          # RAM with queued image buffers
        self._csv_dirty = False           # units arrived since the last CSV write
        self._csv_last = 0.0              # throttle stamp: CSV rewrites at most 1/s
        self._show_pending = None         # unit index to render on the next tick
        self._last_show = 0.0             # render throttle stamp (burst: ~3/s)
        self.worker = None
        self.stop_evt = None
        self._photo = {}             # keep PhotoImage refs alive
        self.zoom = {"overlay": 1.0, "crop": 1.0}    # multiplier over fit-scale
        self.pan = {"overlay": (0.0, 0.0), "crop": (0.0, 0.0)}   # image offset, canvas px
        self._view = {}              # key -> {"w","h"} of last-drawn image
        self._drag = None

        # top-level tabs: Scan (this app) + ROI Editor (draw product zones)
        self.nb_main = ttk.Notebook(root)
        self.nb_main.pack(fill="both", expand=True)
        self.host = ttk.Frame(self.nb_main)
        self.nb_main.add(self.host, text="Scan")
        roi_tab = ttk.Frame(self.nb_main)
        self.nb_main.add(roi_tab, text="ROI Editor")

        self._build_toolbar()
        self._build_body()
        self.template = TemplatePane(roi_tab, root_getter=lambda: self.var_root.get().strip())
        self.template.pack(fill="both", expand=True)
        root.bind("<Key>", self._on_key)
        root.after(100, self._poll)
        root.after(200, self._enable_hwheel)   # trackpad sideways scroll (Win)

    # ---- layout ----
    def _build_toolbar(self):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.host, padding=(4, 4, 4, 0))
        top.pack(fill="x")

        # picture-tree root (holds Fail/ and Image/); pick once per session, the
        # Fail/Image leaves are then derived from Product + Day below.
        ttk.Label(top, text="Root:").pack(side="left")
        self.var_root = tk.StringVar()
        ttk.Entry(top, textvariable=self.var_root, width=34).pack(side="left", padx=2)
        ttk.Button(top, text="Browse…", command=self._pick_root).pack(side="left")

        # mode toggle (Result-mapping needs Fail+Image; Zoning needs Image only)
        ttk.Label(top, text="   Mode:").pack(side="left")
        self.var_mode = tk.StringVar(value="result")
        ttk.Radiobutton(top, text="Result-mapping", value="result",
                        variable=self.var_mode, command=self._mode_changed).pack(side="left")
        ttk.Radiobutton(top, text="Zoning", value="zoning",
                        variable=self.var_mode, command=self._mode_changed).pack(side="left")

        # product : the folder name IS the single identity -- it locates the leaf
        # and (in Zoning) keys the rois.json zone-set.
        ttk.Label(top, text="   Product:").pack(side="left")
        self.var_product = tk.StringVar()
        self.cmb_product = ttk.Combobox(top, textvariable=self.var_product, width=18,
                                        state="readonly")
        self.cmb_product.configure(postcommand=self._refresh_products)
        self.cmb_product.bind("<<ComboboxSelected>>", lambda e: self._product_changed())
        self.cmb_product.pack(side="left", padx=2)

        # ROI zone-set revision (Zoning only)
        ttk.Label(top, text=" Zone version:").pack(side="left")
        self.var_ver = tk.StringVar()
        self.cmb_ver = ttk.Combobox(top, textvariable=self.var_ver, width=12, state="disabled")
        self.cmb_ver.configure(postcommand=self._ver_values)   # re-read on open (editor may have added some)
        self.cmb_ver.pack(side="left")

        # day : click one. Only days carrying an Activescale leaf on the needed
        # side(s) are listed, newest first.
        ttk.Label(top, text="   Day:").pack(side="left")
        self.var_day = tk.StringVar()
        self.lst_days = tk.Listbox(top, height=4, width=12, exportselection=False)
        self.lst_days.pack(side="left", padx=2)
        self.lst_days.bind("<<ListboxSelect>>", lambda e: self._day_changed())

        # read-only echo of the resolved Activescale leaves
        self.var_leaf = tk.StringVar(value="Pick a root, product and day.")
        ttk.Label(self.host, textvariable=self.var_leaf, foreground="grey",
                  padding=(6, 0)).pack(fill="x")

        bar = ttk.Frame(self.host, padding=4)
        bar.pack(fill="x")
        ttk.Label(bar, text="Weights:").pack(side="left")
        self.var_weight = tk.StringVar()
        self.cmb_weight = ttk.Combobox(bar, textvariable=self.var_weight,
                                       values=list(self.weights), width=44, state="readonly")
        if self.weights:
            self.cmb_weight.current(0)
        self.cmb_weight.pack(side="left", padx=2)
        ttk.Button(bar, text="⟳", width=3, command=self._rescan_weights).pack(side="left")

        ttk.Label(bar, text="  Conf:").pack(side="left")
        self.var_conf = tk.DoubleVar(value=self.settings.conf_threshold)
        ttk.Spinbox(bar, from_=0.05, to=0.95, increment=0.05,
                    textvariable=self.var_conf, width=5).pack(side="left")

        self.btn_start = ttk.Button(bar, text="Start", command=self._toggle_watch)
        self.btn_start.pack(side="left", padx=8)
        ttk.Button(bar, text="Criteria…", command=self._settings_dialog).pack(side="left")
        self.var_annot = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Annotations (A)", variable=self.var_annot,
                        command=self._redraw).pack(side="left", padx=6)
        self.var_follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Follow new (F)", variable=self.var_follow).pack(side="left", padx=6)
        ttk.Label(bar, text="  Zoom:").pack(side="left")
        ttk.Button(bar, text="－", width=3, command=lambda: self._zoom_step(1 / 1.2)).pack(side="left")
        ttk.Button(bar, text="＋", width=3, command=lambda: self._zoom_step(1.2)).pack(side="left")
        ttk.Button(bar, text="Fit", width=4, command=self._zoom_reset).pack(side="left")
        self.var_status = tk.StringVar(value="Pick folders and a weight, choose a Mode, then Start.")
        ttk.Label(bar, textvariable=self.var_status).pack(side="left", padx=10)
        self._mode_changed()                     # set initial enabled/disabled fields

    def _both_sides(self):
        """Result-mapping consumes both Fail and Image; Zoning only Image."""
        return self.var_mode.get() != "zoning"

    def _mode_changed(self):
        zoning = self.var_mode.get() == "zoning"
        self.cmb_ver.configure(state="readonly" if zoning else "disabled")
        if not zoning:
            self.var_ver.set("")
        # product/day lists differ per mode (intersection vs Image-only)
        self._refresh_products()
        if zoning and self.var_product.get().strip():
            self._refresh_ver()          # populate revisions for the current product

    def _pick_root(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(
            title="Pick the picture-tree root (the folder holding Fail/ and Image/)")
        if d:
            self.var_root.set(d)
            self._refresh_products()

    def _refresh_products(self):
        root = self.var_root.get().strip()
        vals = pic_products(root, self._both_sides()) if os.path.isdir(root) else []
        self.cmb_product.configure(values=vals)
        if self.var_product.get() not in vals:
            self.var_product.set("")
            self.cmb_ver.configure(values=[])
            self.var_ver.set("")
        self._refresh_days()

    def _product_changed(self):
        self._refresh_days()
        if self.var_mode.get() == "zoning":
            self._refresh_ver()

    def _refresh_ver(self):
        """Populate revisions AND pick a default (active, else newest)."""
        prod = self.var_product.get().strip()
        vers = versions_of(prod) or []
        self.cmb_ver.configure(values=vers)
        act = active_of(prod)
        self.var_ver.set(act if act else (vers[0] if vers else ""))

    def _ver_values(self):
        """postcommand: refresh the revision list on dropdown-open without
        disturbing the current selection (the ROI Editor may have added some)."""
        self.cmb_ver.configure(values=versions_of(self.var_product.get().strip()) or [])

    def _refresh_days(self):
        self.lst_days.delete(0, "end")
        self.var_day.set("")
        root = self.var_root.get().strip()
        prod = self.var_product.get().strip()
        days = pic_days(root, prod, self._both_sides()) if (os.path.isdir(root) and prod) else []
        for d in days:
            self.lst_days.insert("end", d)
        if days:
            self.lst_days.selection_set(0)          # newest first, pre-selected
            self.var_day.set(days[0])
        self._update_leaf_label()

    def _day_changed(self):
        sel = self.lst_days.curselection()
        self.var_day.set(self.lst_days.get(sel[0]) if sel else "")
        self._update_leaf_label()

    def _update_leaf_label(self):
        root = self.var_root.get().strip()
        prod = self.var_product.get().strip()
        day = self.var_day.get().strip()
        if not (root and prod and day):
            self.var_leaf.set("Pick a root, product and day.")
            return
        img = pic_leaf(root, "Image", prod, day)
        if self._both_sides():
            self.var_leaf.set(f"Fail: {pic_leaf(root, 'Fail', prod, day)}   |   Image: {img}")
        else:
            self.var_leaf.set(f"Image: {img}")

    def _build_body(self):
        tk, ttk = self.tk, self.ttk
        pane = ttk.PanedWindow(self.host, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # left: unit list
        left = ttk.Frame(pane)
        pane.add(left, weight=1)
        cols = ("sv", "sug", "human")
        self.tree_units = ttk.Treeview(left, columns=cols, height=25)
        self.tree_units.heading("#0", text="Serial")
        self.tree_units.column("#0", width=130)
        for c, t, w in (("sv", "Station-View", 90), ("sug", "Suggested", 80), ("human", "Human", 80)):
            self.tree_units.heading(c, text=t)
            self.tree_units.column(c, width=w, anchor="center")
        self.tree_units.pack(fill="both", expand=True)
        self.tree_units.bind("<<TreeviewSelect>>", lambda e: self._select_from_tree())
        for v, col in (("NG", "#ffb3b3"), ("Review", "#f0b3f0"),
                       ("Cleaning", "#ffe9a8"), ("Overkill", "#b8e6b8")):
            self.tree_units.tag_configure(v, background=col)

        # right: viewer + detections + verdict bar
        right = ttk.Frame(pane)
        pane.add(right, weight=4)

        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)
        self.cv_overlay = tk.Canvas(self.nb, bg="#222")
        self.nb.add(self.cv_overlay, text="Full-image overlay")
        crop_tab = ttk.Frame(self.nb)
        self.nb.add(crop_tab, text="Crops")
        self.lst_crops = tk.Listbox(crop_tab, width=28)
        self.lst_crops.pack(side="left", fill="y")
        self.lst_crops.bind("<<ListboxSelect>>", lambda e: self._select_crop())
        self.cv_crop = tk.Canvas(crop_tab, bg="#222")
        self.cv_crop.pack(side="left", fill="both", expand=True)
        for cv, key in ((self.cv_overlay, "overlay"), (self.cv_crop, "crop")):
            cv.bind("<Configure>", lambda e: self._redraw())
            # trackpad: two-finger scroll pans the axis that has overflow, so a
            # wide crop scrolls sideways with a plain vertical swipe. Shift forces
            # horizontal; pinch arrives as Ctrl+wheel and zooms. Drag = free pan.
            cv.bind("<MouseWheel>", lambda e, k=key, c=cv: self._scroll(e, k, c, False))
            cv.bind("<Shift-MouseWheel>", lambda e, k=key, c=cv: self._scroll(e, k, c, True))
            cv.bind("<Control-MouseWheel>", lambda e, k=key, c=cv: self._wheel(e, k, c))
            cv.bind("<ButtonPress-1>",
                    lambda e, k=key, c=cv: (c.config(cursor="fleur"), self._pan_start(e, k)))
            cv.bind("<B1-Motion>", lambda e, k=key, c=cv: self._pan_move(e, k, c))
            cv.bind("<ButtonRelease-1>", lambda e, c=cv: c.config(cursor=""))

        det_cols = ("crop", "cls", "conf", "measure", "verdict", "note")
        self.tree_dets = ttk.Treeview(right, columns=det_cols, show="headings", height=6)
        for c, t, w in (("crop", "Crop", 50), ("cls", "Class", 110), ("conf", "Conf", 60),
                        ("measure", "Size", 110), ("verdict", "Verdict", 80), ("note", "Note", 500)):
            self.tree_dets.heading(c, text=t)
            self.tree_dets.column(c, width=w, anchor="w")
        self.tree_dets.pack(fill="x")

        vb = ttk.Frame(right, padding=6)
        vb.pack(fill="x")
        self.var_unit = tk.StringVar(value="")
        ttk.Label(vb, textvariable=self.var_unit, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(vb, text="   Final verdict: ").pack(side="left")
        for i, v in enumerate(VERDICTS, 1):
            ttk.Button(vb, text=f"{i}  {v}",
                       command=lambda vv=v: self._set_verdict(vv)).pack(side="left", padx=3)
        ttk.Label(vb, text="   (keys 1-4; ←/→ = unit; +/-/0 = zoom; drag = pan any way; "
                            "scroll = up/down, Shift+scroll = sideways)").pack(side="left")

    # ---- toolbar actions ----
    def _pick_dir(self, var, title):
        from tkinter import filedialog
        d = filedialog.askdirectory(title=title)
        if d:
            var.set(d)

    def _rescan_weights(self):
        self.weights = find_weights(self.settings.weights_dirs, self.project)
        self.cmb_weight["values"] = list(self.weights)

    def _toggle_watch(self):
        from tkinter import messagebox
        if self.worker and self.worker.is_alive():
            self.stop_evt.set()
            self.btn_start["state"] = "disabled"    # re-enabled by the 'done' message
            self.var_status.set("Stopping after the current image…")
            return
        zoning = self.var_mode.get() == "zoning"
        root = self.var_root.get().strip()
        product = self.var_product.get().strip()
        day = self.var_day.get().strip()
        version = self.var_ver.get().strip()
        if not os.path.isdir(root):
            messagebox.showerror("No root", "Pick the picture-tree root (holds Fail/ and Image/).")
            return
        if not product:
            messagebox.showerror("No product", "Pick a product.")
            return
        if not day:
            messagebox.showerror("No day", "Pick a day.")
            return
        raw_dir = pic_leaf(root, "Image", product, day)
        fail_dir = pic_leaf(root, "Fail", product, day)
        if not os.path.isdir(raw_dir):
            messagebox.showerror("No images", f"No Image Activescale folder:\n{raw_dir}")
            return
        if not zoning and not os.path.isdir(fail_dir):
            messagebox.showerror("No images", f"No Fail Activescale folder:\n{fail_dir}")
            return
        if zoning:
            if not product:
                messagebox.showerror("No product", "Zoning mode: pick a product (draw zones in the ROI Editor first).")
                return
            if not boxes_for(product, version):
                messagebox.showerror(
                    "No zones",
                    f"Product '{product}' / '{version}' has no ROI shapes in:\n{ROI_PATH}\n\n"
                    f"Draw zones in the ROI Editor under the product name '{product}' "
                    f"(open a RAW from this product's Image folder and Save).")
                return
        if not self.var_weight.get():
            messagebox.showerror("No weight", "Pick a trained weight (.pt).")
            return
        self.settings.conf_threshold = float(self.var_conf.get())
        self.units.clear()
        self.current, self.current_crop = None, 0
        self.tree_units.delete(*self.tree_units.get_children())
        self.tree_dets.delete(*self.tree_dets.get_children())
        self.lst_crops.delete(0, "end")
        for cv in (self.cv_overlay, self.cv_crop):
            cv.delete("all")
        self.var_unit.set("")
        self.writer = DateWriter(os.path.join(self.project, self.settings.output_dir))
        already = set()          # always re-evaluate every image in the folder
        self.btn_start["text"] = "Stop"
        self.var_status.set("Loading model…")
        engine = Engine(self.weights[self.var_weight.get()], self.settings)
        self.stop_evt = threading.Event()
        if zoning:
            self.worker = threading.Thread(
                target=self._watch_zoned,
                args=(engine, raw_dir, product, version, already, self.stop_evt), daemon=True)
        else:
            self.worker = threading.Thread(
                target=self._watch, args=(engine, fail_dir, raw_dir, already, self.stop_evt),
                daemon=True)
        self.worker.start()

    def _watch(self, engine, fail_dir, raw_dir, already, stop_evt):
        try:
            engine.load()
        except Exception as e:                                    # noqa: BLE001
            self.q.put(("fatal", f"could not load weight: {e}"))
            return
        skipped = f" ({len(already)} already-recorded file(s) skipped)" if already else ""
        self.q.put(("status", f"Watching {fail_dir} for Fail images…{skipped}"))
        done = set(already)
        sizes = {}                  # basename -> last seen size (copy-finished check)
        raw_missing = {}            # basename -> first time the RAW was missing
        while not stop_evt.is_set():
            for path in sorted(glob.glob(os.path.join(fail_dir, f"*{IMAGE_EXT}"))):
                if stop_evt.is_set():
                    break
                base = os.path.basename(path)
                # errored missing-RAW units stay live: retried until the RAW shows up
                if base in done and base not in raw_missing:
                    continue
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue                    # vanished/renamed mid-scan
                if sizes.get(base) != size:
                    sizes[base] = size          # still being copied: wait one poll
                    continue
                serial, date, station, view = parse_fail_name(path)
                raw_path = find_raw_for(serial, station, view, raw_dir,
                                        fail_stem=os.path.splitext(base)[0])
                if raw_path is None:
                    since = raw_missing.setdefault(base, time.time())
                    if base in done or time.time() - since < RAW_WAIT_S:
                        continue                # the RAW may still be on its way
                    unit = UnitResult(serial, date, station, view, "", path,
                                      error=f"no RAW with serial {serial} in Image folder"
                                            " - retrying while this session runs",
                                      suggested=REVIEW)
                else:
                    raw_missing.pop(base, None)  # a late RAW replaces the error row
                    rec = (serial, date, station, view, raw_path, path)
                    try:
                        unit = engine.process_unit(rec)
                    except Exception as e:                        # noqa: BLE001
                        unit = UnitResult(*rec[:4], rec[4], rec[5],
                                          error=str(e), suggested=REVIEW)
                done.add(base)
                self._worker_save_images(unit)
                self.q.put(("unit", unit))
            stop_evt.wait(self.settings.poll_seconds)
        self.q.put(("done",))

    def _watch_zoned(self, engine, raw_dir, product, version, already, stop_evt):
        """Zoning mode: watch the RAW folder directly; each RAW is one unit cropped
        by the product's ROI zones. The RAW basename is the unit identity."""
        try:
            engine.load()
        except Exception as e:                                    # noqa: BLE001
            self.q.put(("fatal", f"could not load weight: {e}"))
            return
        self.q.put(("status", f"Watching {raw_dir} for RAW images (zoning: {product}/{version})…"))
        done = set(already)
        sizes = {}                  # basename -> last seen size (copy-finished check)
        while not stop_evt.is_set():
            for path in sorted(glob.glob(os.path.join(raw_dir, f"*{IMAGE_EXT}"))):
                if stop_evt.is_set():
                    break
                base = os.path.basename(path)
                if base in done:
                    continue
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue                    # vanished/renamed mid-scan
                if sizes.get(base) != size:
                    sizes[base] = size          # still being copied: wait one poll
                    continue
                try:
                    unit, _ = process_raw(engine, path, self.settings, product, version)
                    if unit is None:            # product lost its zones mid-run
                        continue
                except Exception as e:                            # noqa: BLE001
                    s, d, st, vw = parse_fail_name(path)
                    unit = UnitResult(s, d, st, vw, path, path, error=str(e), suggested=REVIEW)
                done.add(base)
                self._worker_save_images(unit)
                self.q.put(("unit", unit))
            stop_evt.wait(self.settings.poll_seconds)
        self.q.put(("done",))

    def _save_results(self, unit=None):
        """Write CSVs (and one unit's images) tolerating locked/unwritable files
        (e.g. units.csv open in Excel): results stay in memory and the files are
        fully rewritten on the next save. Returns True when everything wrote."""
        try:
            if unit is not None:
                self.writer.save_images(unit)
            self.writer.write_all(self.units,
                                  only_date=None if unit is None else unit.date)
            return True
        except (OSError, cv2.error) as e:
            self.var_status.set(f"Write failed ({e}) - close the file; "
                                "results are rewritten on the next save.")
            return False

    def _worker_save_images(self, unit):
        """Runs on the worker thread. Image encodes are the heavy part of
        persisting a unit; on the GUI thread they starved tk whenever the
        worker (GPU) produced units faster than they could be written --
        whole-burst "Not Responding" with no incremental rows. CSVs stay on
        the GUI thread (they read self.units)."""
        try:
            self.writer.save_images(unit)
        except Exception as e:                                    # noqa: BLE001
            # never let an image write kill the watcher thread
            self.q.put(("status", f"Image write failed ({e}) - images for "
                                  f"{unit.serial} may be missing."))

    def _poll(self):
        # Bounded drain: at most 3 messages per 100ms tick, so tk keeps pumping
        # (live window, rows appearing one by one) even when a GPU-speed worker
        # fills the queue continuously. 3/tick still drains ~30 units/s, far
        # above any producer.
        try:
            for _ in range(3):
                msg = self.q.get_nowait()
                if msg[0] == "unit":
                    unit = msg[1]
                    idx = next((i for i, u in enumerate(self.units)
                                if os.path.basename(u.result_path)
                                == os.path.basename(unit.result_path)), None)
                    # "Follow new" checked = always land on the newest arrival,
                    # even after navigating back (the checkbox -- hotkey F -- is
                    # the one and only control; an invisible "only while parked
                    # at the latest" rule just read as a broken checkbox).
                    # Renders are deferred to _show_pending (throttled below):
                    # rendering every unit of a GPU burst would hog the GUI
                    # thread again.
                    if idx is None:
                        self.units.append(unit)
                        self._add_row(unit)
                        if self.var_follow.get():
                            self._show_pending = len(self.units) - 1
                    else:                       # re-run of a shown unit (late RAW)
                        unit.human, unit.human_time = \
                            self.units[idx].human, self.units[idx].human_time
                        self.units[idx] = unit
                        self._refresh_row(idx)
                        if self.current == idx:
                            self._show_pending = idx
                    self._csv_dirty = True      # images were written by the worker
                    self.var_status.set(f"Processed {len(self.units)}: "
                                        f"{unit.serial} [{unit.suggested}]")
                    if self.current is None and self._show_pending is None:
                        self._show_pending = 0
                    # images are already on disk: keep only the viewed unit's buffers
                    if self.current is not None:
                        self._trim_unit_memory(self.current)
                elif msg[0] == "done":
                    self.btn_start["text"] = "Start"
                    self.btn_start["state"] = "normal"
                    self.var_status.set(f"Stopped - {len(self.units)} unit(s) this session. "
                                        f"Output: {self.writer.root}")
                    self._csv_last = 0.0        # session over: flush the CSVs now
                elif msg[0] == "fatal":
                    self.btn_start["text"] = "Start"
                    self.btn_start["state"] = "normal"
                    self.var_status.set(msg[1])
                elif msg[0] == "status":
                    self.var_status.set(msg[1])
        except queue.Empty:
            pass
        finally:
            # Deferred render: during a burst at most ~3/s; the moment the
            # queue is empty the newest unit renders immediately.
            if self._show_pending is not None and \
                    (self.q.empty() or time.time() - self._last_show >= 0.3):
                self._show_unit(self._show_pending)
                self._last_show = time.time()
            # CSVs are a full merged rewrite (cost grows with the session), so
            # they are batched: at most one write per second, never one per
            # unit. A failed write stays dirty and is retried next second.
            if self._csv_dirty and time.time() - self._csv_last >= 1.0:
                if self._save_results():
                    self._csv_dirty = False
                self._csv_last = time.time()
            self.root.after(100, self._poll)   # re-arm even if a handler raised

    # ---- unit list / display ----
    def _add_row(self, unit):
        self.tree_units.insert("", "end", iid=str(len(self.units) - 1), text=unit.serial,
                               values=(unit.station_view, unit.suggested, unit.human),
                               tags=(unit.suggested,))

    def _refresh_row(self, idx):
        u = self.units[idx]
        self.tree_units.item(str(idx), values=(u.station_view, u.suggested, u.human),
                             tags=(u.human or u.suggested,))

    def _select_from_tree(self):
        sel = self.tree_units.selection()
        if sel:
            self._show_unit(int(sel[0]), from_tree=True)

    def _show_unit(self, idx, from_tree=False):
        self._show_pending = None   # manual navigation cancels any deferred render
        if not (0 <= idx < len(self.units)):
            return
        self.current, self.current_crop = idx, 0
        self.zoom = {"overlay": 1.0, "crop": 1.0}
        self.pan = {"overlay": (0.0, 0.0), "crop": (0.0, 0.0)}
        u = self.units[idx]
        self._rebuild_unit_images(u)     # may have been freed to cap memory
        self._trim_unit_memory(idx)      # keep only this unit's buffers in RAM
        if not from_tree:
            self.tree_units.selection_set(str(idx))
            self.tree_units.see(str(idx))
        total = ("" if u.total_area_mm2 is None else f", total {u.total_area_mm2:.2f}mm²")
        note = f"   [{u.unit_note}]" if u.unit_note else ""
        err = f"   ERROR: {u.error}" if u.error else ""
        peers = [x for x in self.units if x.serial == u.serial and x.date == u.date]
        roll = ""
        if len(peers) > 1:                     # unit disposition across inspections
            humans = [x.human for x in peers if x.human]
            worst = max(humans, key=LADDER.get) if humans else "-"
            roll = f"   serial worst: {worst} ({len(humans)}/{len(peers)} judged)"
        self.var_unit.set(f"{u.serial}  {u.station_view}   suggested: {u.suggested}"
                          f"   ({u.defect_count} defect(s){total}){roll}{note}{err}")
        self.tree_dets.delete(*self.tree_dets.get_children())
        for c in u.crops:
            if not c.detections:
                self.tree_dets.insert("", "end", values=(c.index, "-", "-", "-", c.verdict, c.note))
            for d in c.detections:
                flags = (" !near-limit" if d.near_limit else "") + (" !truncated" if d.truncated else "")
                self.tree_dets.insert("", "end", values=(
                    c.index, d.class_name, f"{d.conf:.2f}", d.measure_text, d.verdict, d.note + flags))
        self.lst_crops.delete(0, "end")
        for c in u.crops:
            self.lst_crops.insert("end", f"crop {c.index} [{c.kind}] -> {c.verdict}")
        if u.crops:
            self.lst_crops.selection_set(0)
        self._redraw()

    def _select_crop(self):
        sel = self.lst_crops.curselection()
        if sel:
            self.current_crop = sel[0]
            self.zoom["crop"], self.pan["crop"] = 1.0, (0.0, 0.0)
            self._redraw()

    def _rebuild_unit_images(self, u):
        """Recreate the display buffers (downscaled overlay + full-res crops) that
        _trim_unit_memory freed, by re-reading the RAW. No-op if still present or
        the RAW is gone/unreadable (an error unit) -- _redraw then just skips."""
        if u.overlay_base is not None or not u.raw_path:
            return
        raw = cv2.imread(u.raw_path)
        if raw is None:
            return
        u.overlay_scale = min(1.0, OVERLAY_MAX_W / raw.shape[1])
        u.overlay_base = cv2.resize(raw, None, fx=u.overlay_scale, fy=u.overlay_scale)
        u.crop_bases = [raw[c.window[1]:c.window[3], c.window[0]:c.window[2]].copy()
                        for c in u.crops]

    def _trim_unit_memory(self, keep_idx):
        """Free the heavy image buffers on every unit except keep_idx. They are
        already written to disk and rebuilt on demand, so session RAM stays ~O(1)
        instead of growing with every processed image."""
        for i, u in enumerate(self.units):
            if i != keep_idx and u.overlay_base is not None:
                u.overlay_base = None
                u.crop_bases = []

    def _redraw(self):
        if self.current is None:
            return
        u = self.units[self.current]
        annotate = self.var_annot.get()
        if u.overlay_base is not None:
            self._blit(self.cv_overlay, render_overlay(u, annotate), "overlay")
        if u.crop_bases and self.current_crop < len(u.crop_bases):
            self._blit(self.cv_crop, render_crop(u, self.current_crop, annotate), "crop")

    def _blit(self, canvas, bgr, key):
        """Draw bgr on canvas at fit-scale * zoom[key], panned by pan[key].
        Only the visible sub-region is resized, so high zoom stays cheap."""
        from PIL import Image, ImageTk
        cw = max(50, canvas.winfo_width())
        ch = max(50, canvas.winfo_height())
        h, w = bgr.shape[:2]
        self._view[key] = {"w": w, "h": h}
        fit = min(cw / w, ch / h)
        s = fit * self.zoom[key]
        dw, dh = w * s, h * s
        ox, oy = self.pan[key]
        x0 = (cw - dw) / 2 + ox                  # image top-left in canvas px
        y0 = (ch - dh) / 2 + oy
        # source pixels visible in the canvas viewport
        vx0 = max(0, int(-x0 / s))
        vy0 = max(0, int(-y0 / s))
        vx1 = min(w, int((cw - x0) / s) + 1)
        vy1 = min(h, int((ch - y0) / s) + 1)
        canvas.delete("all")
        if vx1 <= vx0 or vy1 <= vy0:
            return
        sw = max(1, int((vx1 - vx0) * s))
        sh = max(1, int((vy1 - vy0) * s))
        # nearest keeps pixels crisp when magnified (read fine detail); area
        # averages when shrinking (clean downscale, like the old fit view).
        interp = cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA
        img = cv2.resize(bgr[vy0:vy1, vx0:vx1], (sw, sh), interpolation=interp)
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self._photo[key] = photo
        canvas.create_image(x0 + vx0 * s, y0 + vy0 * s, image=photo, anchor="nw")

    # ---- zoom / pan ----
    def _active_key(self):
        return "crop" if self.nb.index("current") == 1 else "overlay"

    def _active_canvas(self):
        return self.cv_crop if self._active_key() == "crop" else self.cv_overlay

    def _clamp_pan(self, key, cw, ch, fit):
        v = self._view.get(key)
        if not v:
            return
        s = fit * self.zoom[key]
        dw, dh = v["w"] * s, v["h"] * s
        ox, oy = self.pan[key]

        def lim(off, d, c):
            if d <= c:
                return 0.0                       # smaller than canvas: keep centred
            m = (d - c) / 2
            return max(-m, min(m, off))
        self.pan[key] = (lim(ox, dw, cw), lim(oy, dh, ch))

    def _wheel(self, e, key, canvas):
        v = self._view.get(key)
        if self.current is None or not v:
            return
        cw = max(50, canvas.winfo_width())
        ch = max(50, canvas.winfo_height())
        fit = min(cw / v["w"], ch / v["h"])
        old_z = self.zoom[key]
        new_z = min(8.0, max(1.0, old_z * (1.2 if e.delta > 0 else 1 / 1.2)))
        if new_z == old_z:
            return
        s_old, s_new = fit * old_z, fit * new_z
        # keep the image point under the cursor fixed
        x0 = (cw - v["w"] * s_old) / 2 + self.pan[key][0]
        y0 = (ch - v["h"] * s_old) / 2 + self.pan[key][1]
        ix, iy = (e.x - x0) / s_old, (e.y - y0) / s_old
        self.pan[key] = (e.x - ix * s_new - (cw - v["w"] * s_new) / 2,
                         e.y - iy * s_new - (ch - v["h"] * s_new) / 2)
        self.zoom[key] = new_z
        self._clamp_pan(key, cw, ch, fit)
        self._redraw()

    def _scroll(self, e, key, canvas, horizontal):
        """Two-finger trackpad scroll (or wheel) -> pan. Plain scroll pans the
        axis that overflows the canvas (so a wide crop moves sideways with a
        normal vertical swipe); Shift forces horizontal. Snaps back when the
        image fits, via _clamp_pan."""
        v = self._view.get(key)
        if self.current is None or not v:
            return
        cw = max(50, canvas.winfo_width())
        ch = max(50, canvas.winfo_height())
        fit = min(cw / v["w"], ch / v["h"])
        s = fit * self.zoom[key]
        over_x = v["w"] * s - cw                   # >0 -> pannable horizontally
        over_y = v["h"] * s - ch
        step = e.delta / 120 * 60                  # ~60 px per notch
        ox, oy = self.pan[key]
        # no vertical room but horizontal room -> redirect vertical swipe sideways
        if horizontal or (over_y <= 1 and over_x > 1):
            self.pan[key] = (ox + step, oy)
        else:
            self.pan[key] = (ox, oy + step)
        self._clamp_pan(key, cw, ch, fit)
        self._redraw()

    def _hscroll_pixels(self, px):
        """Pan the active canvas horizontally by px (used by the Windows
        horizontal-wheel hook). Safe to call from the WndProc callback."""
        key = self._active_key()
        v = self._view.get(key)
        if self.current is None or not v:
            return
        canvas = self._active_canvas()
        cw = max(50, canvas.winfo_width())
        ch = max(50, canvas.winfo_height())
        ox, oy = self.pan[key]
        self.pan[key] = (ox + px, oy)
        self._clamp_pan(key, cw, ch, min(cw / v["w"], ch / v["h"]))
        self._redraw()

    def _enable_hwheel(self):
        """Tk 8.6 on Windows silently drops WM_MOUSEHWHEEL, so two-finger
        horizontal trackpad scroll never reaches the canvas. Subclass the
        toplevel WndProc and translate it into a horizontal pan. Fully
        fail-safe: any error leaves the app on drag/Shift-scroll panning."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes
            WM_MOUSEHWHEEL = 0x020E
            GWLP_WNDPROC = -4
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            LRESULT = ctypes.c_ssize_t
            WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT,
                                         wintypes.WPARAM, wintypes.LPARAM)
            user32.CallWindowProcW.restype = LRESULT
            user32.CallWindowProcW.argtypes = [WNDPROC, wintypes.HWND, wintypes.UINT,
                                               wintypes.WPARAM, wintypes.LPARAM]
            user32.SetWindowLongPtrW.restype = LRESULT
            user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, LRESULT]
            user32.GetWindowLongPtrW.restype = LRESULT
            user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
            hwnd = self.root.winfo_id()
            old = user32.GetWindowLongPtrW(hwnd, GWLP_WNDPROC)

            def proc(h, msg, wp, lp):
                if msg == WM_MOUSEHWHEEL:
                    try:
                        delta = ctypes.c_short((wp >> 16) & 0xFFFF).value
                        self._hscroll_pixels(-delta / 120 * 60)
                    except Exception:
                        pass
                    return 0
                return user32.CallWindowProcW(self._old_wndproc, h, msg, wp, lp)

            self._old_wndproc = WNDPROC(old)          # keep alive
            self._new_wndproc = WNDPROC(proc)         # keep alive
            user32.SetWindowLongPtrW(
                hwnd, GWLP_WNDPROC,
                ctypes.cast(self._new_wndproc, ctypes.c_void_p).value)
        except Exception:
            pass                                      # keep drag/Shift fallback

    def _zoom_step(self, factor):
        key, canvas = self._active_key(), self._active_canvas()
        cw = max(50, canvas.winfo_width())
        ch = max(50, canvas.winfo_height())
        e = type("E", (), {"x": cw / 2, "y": ch / 2, "delta": 1 if factor > 1 else -1})()
        self._wheel(e, key, canvas)

    def _zoom_reset(self, key=None):
        key = key or self._active_key()
        self.zoom[key] = 1.0
        self.pan[key] = (0.0, 0.0)
        self._redraw()

    def _pan_start(self, e, key):
        self._drag = (e.x, e.y, self.pan[key])

    def _pan_move(self, e, key, canvas):
        if not self._drag:
            return
        sx, sy, (ox, oy) = self._drag
        self.pan[key] = (ox + (e.x - sx), oy + (e.y - sy))
        cw = max(50, canvas.winfo_width())
        ch = max(50, canvas.winfo_height())
        v = self._view.get(key)
        if v:
            self._clamp_pan(key, cw, ch, min(cw / v["w"], ch / v["h"]))
        self._redraw()

    # ---- verdicts / keys ----
    def _set_verdict(self, v):
        if self.current is None:
            return
        u = self.units[self.current]
        u.human, u.human_time = v, time.strftime("%Y-%m-%d %H:%M:%S")
        self._refresh_row(self.current)
        self._save_results()
        for i in range(self.current + 1, len(self.units)):      # advance to next unjudged
            if not self.units[i].human:
                self._show_unit(i)
                return
        self.var_status.set("All shown units have a human verdict.")

    def _on_key(self, e):
        if self.nb_main.index("current") != 0:   # review keys act only on the Scan tab
            return
        if isinstance(e.widget, (self.tk.Entry, self.tk.Spinbox, self.tk.Text)):
            return
        if e.char and e.char in "1234":   # arrows/modifiers have e.char == ""
            self._set_verdict(VERDICTS[int(e.char) - 1])
        elif e.keysym.lower() == "a":
            self.var_annot.set(not self.var_annot.get())
            self._redraw()
        elif e.keysym == "Right" and self.current is not None:
            self._show_unit(self.current + 1)
        elif e.keysym == "Left" and self.current is not None:
            self._show_unit(self.current - 1)
        elif e.keysym in ("plus", "KP_Add", "equal"):
            self._zoom_step(1.2)
        elif e.keysym in ("minus", "KP_Subtract"):
            self._zoom_step(1 / 1.2)
        elif e.keysym in ("0", "KP_0"):
            self._zoom_reset()
        elif e.keysym.lower() == "f":
            self.var_follow.set(not self.var_follow.get())

    # ---- criteria dialog (session-only; config.yaml stays the source of truth) ----
    def _settings_dialog(self):
        tk, ttk = self.tk, self.ttk
        s = self.settings
        top = tk.Toplevel(self.root)
        top.title("Size criteria & calibration (session-only)")
        top.grab_set()
        frm = ttk.Frame(top, padding=10)
        frm.pack(fill="both", expand=True)

        fields = [
            ("Scratch length limit (mm)", s.scratch_length_mm),
            ("Scuffmark area limit (mm²)", s.scuffmark_area_mm2),
            ("Near-limit margin (0-1)", s.near_limit_margin),
            ("Unit max defect count", s.max_defect_count),
            ("Unit max total area (mm²)", s.max_total_area_mm2),
            ("Default mm per pixel", s.default_mm_per_px),
        ]
        vars_ = []
        for r, (label, val) in enumerate(fields):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=2)
            v = tk.StringVar(value="" if val is None else str(val))
            ttk.Entry(frm, textvariable=v, width=12).grid(row=r, column=1, padx=6)
            vars_.append(v)
        ttk.Label(frm, text="mm/px per station-view (one 'STATION-VIEW: value' per line)")\
            .grid(row=len(fields), column=0, columnspan=2, sticky="w", pady=(8, 2))
        txt = tk.Text(frm, width=40, height=5)
        txt.grid(row=len(fields) + 1, column=0, columnspan=2, sticky="we")
        txt.insert("1.0", "\n".join(f"{k}: {v}" for k, v in s.per_station_view.items()))
        ttk.Label(frm, foreground="#666",
                  text="Applies to this session only and re-judges processed units.\n"
                       "For permanent values edit config.yaml [rejudge:].")\
            .grid(row=len(fields) + 2, column=0, columnspan=2, sticky="w", pady=6)

        def apply():
            def num(v):
                v = v.get().strip()
                return float(v) if v else None
            s.scratch_length_mm = num(vars_[0])
            s.scuffmark_area_mm2 = num(vars_[1])
            s.near_limit_margin = num(vars_[2]) or 0.8
            c = num(vars_[3])
            s.max_defect_count = int(c) if c is not None else None
            s.max_total_area_mm2 = num(vars_[4])
            s.default_mm_per_px = num(vars_[5])
            s.per_station_view = {}
            for line in txt.get("1.0", "end").splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    try:
                        s.per_station_view[k.strip()] = float(v)
                    except ValueError:
                        pass
            for u in self.units:                   # re-judge from stored px measures
                judge_unit(u, s)
            for i in range(len(self.units)):
                self._refresh_row(i)
            if self.writer:
                self._save_results()
            if self.current is not None:
                self._show_unit(self.current)
            top.destroy()

        ttk.Button(frm, text="Apply", command=apply)\
            .grid(row=len(fields) + 3, column=0, columnspan=2, pady=4)




# ================================================= zoning geometry + ROI store ==
# (copied from surface_scan.py)

BOX_SIZE = 640                                  # default square crop == model imgsz
HALF = BOX_SIZE / 2
ROI_PATH = os.path.join(app_dir(), "rois.json")
MIN_SIDE = 8.0                                   # smallest rect side / poly extent (native px)
CLICK_TOL = 6.0                                  # drag under this (canvas px) counts as a click


# ------------------------------------------------------------------ ROI shapes --
# An ROI is a plain-JSON dict so it round-trips through rois.json. Legacy boxes
# were bare [cx, cy] squares of BOX_SIZE. Shape types:
#   {"type":"rect",  "cx","cy","w","h"}       (square == rect with w == h)
#   {"type":"circle","cx","cy","r"}
#   {"type":"poly",  "pts":[[x,y], ...]}
# The model always eats a rectangular crop = the shape's bounding box; non-rect
# shapes get everything outside the shape zeroed so the net only sees the region.

def to_shape(b, box_size=BOX_SIZE):
    """Coerce a stored box (legacy [cx,cy] OR a shape dict) into a shape dict."""
    if isinstance(b, dict):
        t = b.get("type", "rect")
        if t == "circle":
            return {"type": "circle", "cx": float(b["cx"]), "cy": float(b["cy"]),
                    "r": float(b["r"])}
        if t == "poly":
            return {"type": "poly", "pts": [[float(x), float(y)] for x, y in b["pts"]]}
        return {"type": "rect", "cx": float(b["cx"]), "cy": float(b["cy"]),
                "w": float(b.get("w", box_size)), "h": float(b.get("h", box_size))}
    cx, cy = b
    return {"type": "rect", "cx": float(cx), "cy": float(cy),
            "w": float(box_size), "h": float(box_size)}


def _pt_seg_dist(px, py, ax, ay, bx, by):
    """Distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * dx, ay + t * dy
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5


def shape_bbox(sh):
    """(x0, y0, x1, y1) native-px bounding box, floats."""
    t = sh["type"]
    if t == "circle":
        return (sh["cx"] - sh["r"], sh["cy"] - sh["r"],
                sh["cx"] + sh["r"], sh["cy"] + sh["r"])
    if t == "poly":
        xs = [p[0] for p in sh["pts"]]
        ys = [p[1] for p in sh["pts"]]
        return (min(xs), min(ys), max(xs), max(ys))
    hw, hh = sh["w"] / 2, sh["h"] / 2
    return (sh["cx"] - hw, sh["cy"] - hh, sh["cx"] + hw, sh["cy"] + hh)


def shape_center(sh):
    x0, y0, x1, y1 = shape_bbox(sh)
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def shape_move(sh, dx, dy):
    if sh["type"] == "poly":
        return {"type": "poly", "pts": [[x + dx, y + dy] for x, y in sh["pts"]]}
    out = dict(sh)
    out["cx"] = sh["cx"] + dx
    out["cy"] = sh["cy"] + dy
    return out


def shape_contains(sh, x, y):
    t = sh["type"]
    if t == "circle":
        return (x - sh["cx"]) ** 2 + (y - sh["cy"]) ** 2 <= sh["r"] ** 2
    if t == "poly":
        return cv2.pointPolygonTest(np.array(sh["pts"], np.float32),
                                    (float(x), float(y)), False) >= 0
    hw, hh = sh["w"] / 2, sh["h"] / 2
    return abs(x - sh["cx"]) <= hw and abs(y - sh["cy"]) <= hh


def shape_mask(sh, ox, oy, w, h):
    """uint8 (h, w) mask, 255 inside the shape, for a crop whose top-left is
    (ox, oy) in native px. Rect returns all-255 (no masking needed)."""
    m = np.zeros((h, w), np.uint8)
    t = sh["type"]
    if t == "circle":
        cv2.circle(m, (int(round(sh["cx"] - ox)), int(round(sh["cy"] - oy))),
                   max(1, int(round(sh["r"]))), 255, -1)
    elif t == "poly":
        pts = np.array([[int(round(x - ox)), int(round(y - oy))]
                        for x, y in sh["pts"]], np.int32)
        cv2.fillPoly(m, [pts], 255)
    else:
        m[:] = 255
    return m


def clip_poly_to_shape(poly, sh):
    """Intersect a detection polygon (Nx2 native px) with the ROI shape. Returns
    the largest in-region contour (Nx2 float32) or None if nothing is inside.
    Guarantees a reported detection never extends past the drawn region."""
    x0, y0, x1, y1 = shape_bbox(sh)
    ox, oy = int(np.floor(x0)), int(np.floor(y0))
    w, h = int(np.ceil(x1)) - ox, int(np.ceil(y1)) - oy
    if w <= 0 or h <= 0:
        return None
    region = shape_mask(sh, ox, oy, w, h)               # 255 inside the shape
    dm = np.zeros((h, w), np.uint8)
    pts = np.round(np.asarray(poly, np.float32) - [ox, oy]).astype(np.int32)
    cv2.fillPoly(dm, [pts], 255)
    inter = cv2.bitwise_and(region, dm)
    cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 1.0:
        return None
    return c.reshape(-1, 2).astype(np.float32) + [ox, oy]


def shape_handles(sh):
    """[(handle_id, x, y)] native-px resize grips for the selected shape."""
    t = sh["type"]
    if t == "circle":
        return [("r", sh["cx"] + sh["r"], sh["cy"])]
    if t == "poly":
        return [(i, x, y) for i, (x, y) in enumerate(sh["pts"])]
    hw, hh = sh["w"] / 2, sh["h"] / 2
    cx, cy = sh["cx"], sh["cy"]
    return [("nw", cx - hw, cy - hh), ("ne", cx + hw, cy - hh),
            ("sw", cx - hw, cy + hh), ("se", cx + hw, cy + hh)]


def shape_resize(sh, handle, x, y):
    """New shape after dragging `handle` to native (x, y)."""
    t = sh["type"]
    if t == "circle":
        r = max(MIN_SIDE, ((x - sh["cx"]) ** 2 + (y - sh["cy"]) ** 2) ** 0.5)
        out = dict(sh)
        out["r"] = r
        return out
    if t == "poly":
        pts = [list(p) for p in sh["pts"]]
        pts[handle] = [x, y]
        return {"type": "poly", "pts": pts}
    x0, y0, x1, y1 = shape_bbox(sh)
    fx, fy = {"nw": (x1, y1), "ne": (x0, y1),
              "sw": (x1, y0), "se": (x0, y0)}[handle]     # opposite corner stays put
    nx0, nx1 = min(fx, x), max(fx, x)
    ny0, ny1 = min(fy, y), max(fy, y)
    w = max(MIN_SIDE, nx1 - nx0)
    h = max(MIN_SIDE, ny1 - ny0)
    return {"type": "rect", "cx": (nx0 + nx1) / 2, "cy": (ny0 + ny1) / 2, "w": w, "h": h}


def shape_clamp(sh, img_w, img_h):
    """Slide the shape so its bbox stays inside the image (centre if it doesn't fit)."""
    x0, y0, x1, y1 = shape_bbox(sh)
    dx = dy = 0.0
    if (x1 - x0) <= img_w:
        dx = -x0 if x0 < 0 else (img_w - x1 if x1 > img_w else 0.0)
    else:
        dx = (img_w - (x0 + x1)) / 2
    if (y1 - y0) <= img_h:
        dy = -y0 if y0 < 0 else (img_h - y1 if y1 > img_h else 0.0)
    else:
        dy = (img_h - (y0 + y1)) / 2
    return shape_move(sh, dx, dy) if (dx or dy) else sh


def shape_json(sh):
    """Round to ints for compact, stable rois.json storage."""
    t = sh["type"]
    if t == "circle":
        return {"type": "circle", "cx": int(round(sh["cx"])), "cy": int(round(sh["cy"])),
                "r": int(round(sh["r"]))}
    if t == "poly":
        return {"type": "poly", "pts": [[int(round(x)), int(round(y))] for x, y in sh["pts"]]}
    return {"type": "rect", "cx": int(round(sh["cx"])), "cy": int(round(sh["cy"])),
            "w": int(round(sh["w"])), "h": int(round(sh["h"]))}


# ------------------------------------------- template store (products/versions) --

def suggest_product(path):
    """Filename-based product suggestion '{station}-{view}' -- only a default the
    user may keep or overwrite; box lookup never depends on it."""
    _, _, station, view = parse_fail_name(path)
    return f"{station}-{view}"


def load_rois(path=ROI_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}


def products(path=ROI_PATH):
    return sorted(load_rois(path))


def _normalize(entry):
    """(box_size, active_name, versions_dict) for a product, upgrading the old flat
    {reference_raw, box_size, boxes} form to a single 'v1' version."""
    if not entry:
        return BOX_SIZE, None, {}
    if "versions" in entry:
        return entry.get("box_size", BOX_SIZE), entry.get("active"), dict(entry["versions"])
    vers = {"v1": {"reference_raw": entry.get("reference_raw", ""),
                   "boxes": entry.get("boxes", [])}}
    return entry.get("box_size", BOX_SIZE), "v1", vers


def versions_of(product, path=ROI_PATH):
    _bs, _a, vers = _normalize(load_rois(path).get(product))
    return list(vers.keys())


def active_of(product, path=ROI_PATH):
    _bs, active, vers = _normalize(load_rois(path).get(product))
    return active if active in vers else next(iter(vers), None)


def boxes_for(product, version=None, path=ROI_PATH):
    """Shape dicts for a product's version; falls back to its active version,
    then its first, then []. Legacy [cx,cy] boxes migrate to squares here so an
    unknown version never crashes a scan."""
    box_size, active, vers = _normalize(load_rois(path).get(product))
    name = version if (version and version in vers) else \
        (active if active in vers else next(iter(vers), None))
    if name is None:
        return []
    return [to_shape(b, box_size) for b in vers[name].get("boxes", [])]


def _write(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def save_version(product, version, boxes, reference_raw, path=ROI_PATH, make_active=True):
    data = load_rois(path)
    box_size, active, vers = _normalize(data.get(product))
    vers[version] = {"reference_raw": reference_raw,
                     "boxes": [shape_json(s) for s in boxes]}
    if make_active or active not in vers:
        active = version
    data[product] = {"box_size": box_size, "active": active, "versions": vers}
    _write(data, path)


def delete_version(product, version, path=ROI_PATH):
    data = load_rois(path)
    box_size, active, vers = _normalize(data.get(product))
    vers.pop(version, None)
    if not vers:
        data.pop(product, None)
    else:
        if active not in vers:
            active = next(iter(vers))
        data[product] = {"box_size": box_size, "active": active, "versions": vers}
    _write(data, path)


# ------------------------------------------------------------------ engine --

_ILLEGAL = set('<>:"/\\|?*')


def _safe(s):
    """Filesystem-safe token: parse_fail_name yields '?' for missing station/view
    and arbitrary filenames may carry illegal chars -- output paths must survive
    both now that the boxes no longer depend on a canonical filename."""
    out = "".join("_" if ch in _ILLEGAL else ch for ch in str(s)).strip()
    return out or "na"




def _tile_starts(z0, z1, limit):
    """Start offsets of CROP_SIZE-wide tiles covering [z0, z1), clamped to the
    image [0, limit). Neighbours overlap by TILE_OVERLAP; the last tile is laid
    flush with the far edge so coverage never falls short."""
    lo = max(0, min(int(z0), limit - CROP_SIZE))
    hi = max(lo, min(int(z1), limit) - CROP_SIZE)
    if hi == lo:
        return [lo]
    starts = list(range(lo, hi, CROP_SIZE - TILE_OVERLAP))
    starts.append(hi)
    return starts


def _merge_dets(dets, owners):
    """Merge duplicate/fragment detections of the same class whose masks overlap
    or nearly touch (<= PAD px apart) -- what overlapping tiles and adjacent
    zones produce for one physical defect. Returns [(merged Detection, owner of
    its highest-conf member)]. Pairwise raster tests are O(n^2) but n is a
    handful of defects, not thousands."""
    PAD = 4                    # zone-clipped fragments touch but don't overlap
    n = len(dets)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    boxes = [d.bbox() for d in dets]
    kernel = np.ones((2 * PAD + 1, 2 * PAD + 1), np.uint8)
    for a in range(n):
        for b in range(a + 1, n):
            if dets[a].class_name != dets[b].class_name:
                continue
            ax0, ay0, ax1, ay1 = boxes[a]
            bx0, by0, bx1, by1 = boxes[b]
            if (ax0 > bx1 + PAD or bx0 > ax1 + PAD
                    or ay0 > by1 + PAD or by0 > ay1 + PAD):
                continue                              # bboxes don't even touch
            ox, oy = int(min(ax0, bx0)) - PAD, int(min(ay0, by0)) - PAD
            w = int(max(ax1, bx1)) - ox + 2 * PAD
            h = int(max(ay1, by1)) - oy + 2 * PAD
            ma = np.zeros((h, w), np.uint8)
            mb = np.zeros((h, w), np.uint8)
            cv2.fillPoly(ma, [np.round(dets[a].poly - [ox, oy]).astype(np.int32)], 255)
            cv2.fillPoly(mb, [np.round(dets[b].poly - [ox, oy]).astype(np.int32)], 255)
            if cv2.bitwise_and(cv2.dilate(ma, kernel), mb).any():
                parent[find(a)] = find(b)

    out = []
    for root in {find(i) for i in range(n)}:
        idxs = [i for i in range(n) if find(i) == root]
        best = max(idxs, key=lambda i: dets[i].conf)
        if len(idxs) > 1:      # union raster -> one contour spanning the seams
            gx0 = min(boxes[i][0] for i in idxs)
            gy0 = min(boxes[i][1] for i in idxs)
            gx1 = max(boxes[i][2] for i in idxs)
            gy1 = max(boxes[i][3] for i in idxs)
            ox, oy = int(gx0) - PAD, int(gy0) - PAD
            w, h = int(gx1) - ox + 2 * PAD, int(gy1) - oy + 2 * PAD
            m = np.zeros((h, w), np.uint8)
            for i in idxs:
                cv2.fillPoly(m, [np.round(dets[i].poly - [ox, oy]).astype(np.int32)], 255)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                dets[best].poly = c.reshape(-1, 2).astype(np.float32) + [ox, oy]
        out.append((dets[best], owners[best]))
    return out


# ================================================== zoning crop (process_raw) ==
def process_raw(engine, raw_path, settings, product, version=None):
    """(UnitResult or None, product). None == the product/version has no boxes.
    The filename is parsed only to file the result. Read failure -> Review unit."""
    serial, date, station, view = parse_fail_name(raw_path)
    serial, station, view = _safe(serial), _safe(station), _safe(view)
    boxes = boxes_for(product, version)
    if not boxes:
        return None, product

    raw = cv2.imread(raw_path)
    if raw is None:
        return (UnitResult(serial, date, station, view, raw_path, raw_path,
                           error="could not read RAW", suggested=REVIEW), product)

    raw_h, raw_w = raw.shape[:2]
    unit = UnitResult(serial, date, station, view, raw_path, raw_path)

    # Tiled inference (SAHI-style). Every zone is covered by CROP_SIZE-square
    # tiles at NATIVE resolution, TILE_OVERLAP px of overlap: the model always
    # sees the scale it was trained on (same window as crop mode) -- never a
    # letterbox-shrunken whole zone and never synthetic black mask edges (the
    # old bitwise_and masking was the false-positive source; out-of-zone
    # detections are removed by clipping AFTER prediction instead).
    crops, tiles, owner_of_tile = [], [], []
    for i, shape in enumerate(boxes):
        bx0, by0, bx1, by1 = shape_bbox(shape)
        x0 = max(0, int(round(bx0)))
        y0 = max(0, int(round(by0)))
        x1 = min(raw_w, int(round(bx1)))
        y1 = min(raw_h, int(round(by1)))
        if x1 <= x0 or y1 <= y0:
            continue                                   # shape fully off-image
        for ty in _tile_starts(y0, y1, raw_h):
            for tx in _tile_starts(x0, x1, raw_w):
                tiles.append((tx, ty))
                owner_of_tile.append(len(crops))
        crops.append(CropResult(i, "roi", (x0, y0, x1, y1), shape=shape))
    subs = [raw[ty:ty + CROP_SIZE, tx:tx + CROP_SIZE] for tx, ty in tiles]

    # Clip each tile's detections to its zone; a tile sees native context past
    # the zone edge, and neighbouring zones find their own copy via their own
    # tiles -- _merge_dets then collapses tile-seam fragments and cross-zone
    # duplicates of the same physical defect into one Detection.
    flat, flat_owner = [], []
    for (tx, ty), k, dets in zip(tiles, owner_of_tile, engine.predict_many(subs)):
        for name, conf, local in dets:
            det = Detection(name, conf, local + np.array([tx, ty], np.float32))
            clipped = clip_poly_to_shape(det.poly, crops[k].shape)
            if clipped is None:
                continue                               # detection lies fully outside the ROI
            det.poly = clipped                         # region boundary is deliberate: no _refine
            flat.append(det)
            flat_owner.append(k)

    for det, k in _merge_dets(flat, flat_owner):
        cx, cy = det.poly.mean(axis=0)                 # merged det spanning two zones is
        k = next((j for j, c in enumerate(crops)       # filed under its centroid's zone
                  if shape_contains(c.shape, cx, cy)), k)
        det.length_px, det.area_px = measure_poly(det.poly)
        crops[k].detections.append(det)
    unit.crops.extend(crops)

    judge_unit(unit, settings)
    unit.overlay_scale = min(1.0, OVERLAY_MAX_W / raw_w)
    unit.overlay_base = cv2.resize(raw, None, fx=unit.overlay_scale, fy=unit.overlay_scale)
    unit.crop_bases = [raw[c.window[1]:c.window[3], c.window[0]:c.window[2]].copy()
                       for c in unit.crops]
    return unit, product


# --------------------------------------------------------- template drawing --



# ============================================================ ROI editor tab ==
class TemplatePane(ttk.Frame):
    """Pan/zoom reference view with Draw and Select tools; boxes saved under a
    product name (versioned)."""

    def __init__(self, master, root_getter=None):
        super().__init__(master, padding=4)
        self._root_getter = root_getter      # Scan tab's picture-tree root (for product names)
        self.raw_path = ""
        self.img = None                  # full-res BGR reference (for blit)
        self.img_w = self.img_h = 0
        self.zoom = 1.0
        self.pan = [0.0, 0.0]
        self.boxes = []                  # list of shape dicts (see to_shape)
        self.selected = None
        self.version = ""
        self._drag = None
        self._poly = []                  # vertices of an in-progress polygon (native px)
        self._sel_vertex = None          # index of the picked vertex on the selected polygon
        self.boxes_preview = None        # live rect/circle being dragged out
        self.photo = None

        self.var_tool = tk.StringVar(value="draw")
        self.var_shape = tk.StringVar(value="rect")
        self.var_product = tk.StringVar()
        self.var_version = tk.StringVar()

        # toolbar : labeled groups, two rows ------------------------------
        bar = ttk.Frame(self)
        bar.pack(fill="x")

        g_file = ttk.Labelframe(bar, text="File", padding=(6, 2))
        g_file.pack(side="left", padx=(0, 6))
        ttk.Button(g_file, text="Open reference RAW…", command=self.open_raw).pack(side="left")

        g_tool = ttk.Labelframe(bar, text="Tool", padding=(6, 2))
        g_tool.pack(side="left", padx=6)
        ttk.Radiobutton(g_tool, text="Draw", value="draw", variable=self.var_tool,
                        command=self._tool_changed).pack(side="left")
        ttk.Radiobutton(g_tool, text="Select", value="select", variable=self.var_tool,
                        command=self._tool_changed).pack(side="left")

        g_shape = ttk.Labelframe(bar, text="Shape  (Draw)", padding=(6, 2))
        g_shape.pack(side="left", padx=6)
        for txt, val in (("Rect", "rect"), ("Circle", "circle"), ("Polygon", "poly")):
            ttk.Radiobutton(g_shape, text=txt, value=val, variable=self.var_shape,
                            command=self._shape_changed).pack(side="left")

        g_view = ttk.Labelframe(bar, text="View", padding=(6, 2))
        g_view.pack(side="left", padx=6)
        ttk.Button(g_view, text="－", width=3, command=lambda: self._zoom_step(1 / 1.2)).pack(side="left")
        ttk.Button(g_view, text="＋", width=3, command=lambda: self._zoom_step(1.2)).pack(side="left")
        ttk.Button(g_view, text="Fit", width=4, command=self._fit).pack(side="left", padx=(2, 0))

        g_nudge = ttk.Labelframe(bar, text="Nudge  (← → ↑ ↓, Shift = ×10)", padding=(6, 2))
        g_nudge.pack(side="left", padx=6)
        ttk.Label(g_nudge, text="step").pack(side="left")
        self.var_step = tk.IntVar(value=1)
        self.spn_step = ttk.Spinbox(g_nudge, from_=1, to=999, width=4, textvariable=self.var_step)
        self.spn_step.pack(side="left", padx=(4, 2))
        ttk.Label(g_nudge, text="px").pack(side="left")
        # commit + hand focus back to canvas so arrows resume moving boxes
        self.spn_step.bind("<Return>", lambda e: self.canvas.focus_set())
        self.spn_step.bind("<Escape>", lambda e: self.canvas.focus_set())

        bar2 = ttk.Frame(self)
        bar2.pack(fill="x", pady=(4, 0))

        g_prod = ttk.Labelframe(bar2, text="Product / Version", padding=(6, 2))
        g_prod.pack(side="left", padx=(0, 6))
        self.cmb_product = ttk.Combobox(g_prod, textvariable=self.var_product, width=20)
        self.cmb_product.pack(side="left")
        self.cmb_product.configure(postcommand=self._refresh_products)
        self.cmb_product.bind("<<ComboboxSelected>>", lambda e: self._load_product())
        self.cmb_product.bind("<Return>", lambda e: self._load_product(only_if_exists=True))
        self.cmb_version = ttk.Combobox(g_prod, textvariable=self.var_version, width=14,
                                        state="readonly")
        self.cmb_version.pack(side="left", padx=(6, 2))
        self.cmb_version.bind("<<ComboboxSelected>>", lambda e: self._switch_version())
        ttk.Button(g_prod, text="New ver…", command=self._new_version).pack(side="left", padx=2)
        ttk.Button(g_prod, text="Delete ver", command=self._delete_version).pack(side="left")

        g_act = ttk.Labelframe(bar2, text="Boxes", padding=(6, 2))
        g_act.pack(side="left", padx=6)
        ttk.Button(g_act, text="Save", command=self.save).pack(side="left")
        ttk.Button(g_act, text="Clear", command=self.clear).pack(side="left", padx=(4, 0))

        self.var_status = tk.StringVar(value=f"Open a reference RAW. Box = {BOX_SIZE}px.")
        ttk.Label(bar2, textvariable=self.var_status, anchor="w").pack(
            side="left", fill="x", expand=True, padx=12)

        # canvas -----------------------------------------------------------
        self.canvas = tk.Canvas(self, bg="#202020", highlightthickness=0,
                                cursor=self._cursor())
        self.canvas.pack(fill="both", expand=True, pady=4)
        c = self.canvas
        c.bind("<Configure>", lambda e: self.redraw())
        c.bind("<Motion>", self.on_motion)
        c.bind("<Button-1>", self.on_press)
        c.bind("<Double-Button-1>", self.on_double)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Button-3>", self.on_right)
        c.bind("<ButtonPress-2>", self._mid_pan_start)
        c.bind("<B2-Motion>", self.on_drag)
        c.bind("<ButtonRelease-2>", self.on_release)
        c.bind("<MouseWheel>", lambda e: self._wheel_pan(e, horizontal=False))
        c.bind("<Shift-MouseWheel>", lambda e: self._wheel_pan(e, horizontal=True))
        c.bind("<Control-MouseWheel>", self._wheel_zoom)
        top = master.winfo_toplevel()
        top.bind("<Delete>", self.delete_selected)
        top.bind("<BackSpace>", self._backspace)
        top.bind("<Control-z>", lambda e: self._undo_poly_point())
        top.bind("<Left>", lambda e: self.nudge_selected(-1, 0))
        top.bind("<Right>", lambda e: self.nudge_selected(1, 0))
        top.bind("<Up>", lambda e: self.nudge_selected(0, -1))
        top.bind("<Down>", lambda e: self.nudge_selected(0, 1))
        top.bind("<Shift-Left>", lambda e: self.nudge_selected(-1, 0, mult=10))
        top.bind("<Shift-Right>", lambda e: self.nudge_selected(1, 0, mult=10))
        top.bind("<Shift-Up>", lambda e: self.nudge_selected(0, -1, mult=10))
        top.bind("<Shift-Down>", lambda e: self.nudge_selected(0, 1, mult=10))
        top.bind("<Return>", lambda e: self._finish_poly())
        top.bind("<Escape>", lambda e: self._cancel_poly())

    # ---- transform --------------------------------------------------------
    def _tf(self):
        cw = max(50, self.canvas.winfo_width())
        ch = max(50, self.canvas.winfo_height())
        fit = min(cw / self.img_w, ch / self.img_h) if self.img_w else 1.0
        s = fit * self.zoom
        x0 = (cw - self.img_w * s) / 2 + self.pan[0]
        y0 = (ch - self.img_h * s) / 2 + self.pan[1]
        return s, x0, y0, cw, ch

    def n2c(self, nx, ny, tf):
        s, x0, y0, _, _ = tf
        return x0 + nx * s, y0 + ny * s

    def c2n(self, px, py, tf):
        s, x0, y0, _, _ = tf
        return (px - x0) / s, (py - y0) / s

    def clamp_shape(self, sh):
        return shape_clamp(sh, self.img_w, self.img_h)

    def box_at(self, nx, ny):
        for i in reversed(range(len(self.boxes))):
            if shape_contains(self.boxes[i], nx, ny):
                return i
        return None

    def handle_at(self, nx, ny, tf):
        """(handle_id) of the selected shape's grip under the cursor, else None."""
        if self.selected is None or self.selected >= len(self.boxes):
            return None
        s = tf[0]
        tol = 7 / s                                    # ~7 canvas px in native units
        for hid, hx, hy in shape_handles(self.boxes[self.selected]):
            if abs(nx - hx) <= tol and abs(ny - hy) <= tol:
                return hid
        return None

    def tool_draw(self):
        return self.var_tool.get() == "draw"

    def _cursor(self):
        if self.tool_draw():
            return "crosshair"
        return "arrow"

    def _tool_changed(self):
        self._cancel_poly()
        self.canvas.config(cursor=self._cursor())
        self.redraw()

    def _shape_changed(self):
        self._cancel_poly()
        self.canvas.config(cursor=self._cursor())
        self.redraw()

    # ---- product / version / file ----------------------------------------
    def _refresh_products(self):
        """Offer the SAME product identity the scan uses: the folder names under
        the picture-tree root (Image side), unioned with any product already in
        rois.json (so existing/legacy zone-sets stay reachable)."""
        names = set(products())
        root = self._root_getter() if self._root_getter else ""
        if root and os.path.isdir(root):
            names |= set(pic_products(root, False))
        self.cmb_product["values"] = sorted(names)

    def _refresh_versions(self, product, select=None):
        names = versions_of(product) or ["v1"]
        self.cmb_version["values"] = names
        pick = select or active_of(product) or names[0]
        if pick not in names:
            pick = names[0]
        self.version = pick
        self.var_version.set(pick)

    def _load_product(self, only_if_exists=False):
        """Load an existing product's versions + boxes. Typing a NEW name and
        pressing Enter keeps the current boxes (you're starting that product)."""
        prod = self.var_product.get().strip()
        if only_if_exists and prod not in load_rois():
            return
        self._refresh_versions(prod)
        self.boxes = list(boxes_for(prod, self.version))
        self.selected = None
        self._poly = []
        self.redraw()

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Bad image", f"Could not read:\n{path}")
            return False
        self.raw_path = path
        self.img = img
        self.img_h, self.img_w = img.shape[:2]
        self.zoom, self.pan, self.selected = 1.0, [0.0, 0.0], None
        self._refresh_products()
        # If the RAW came from .../Image/{product}/{day}/Activescale, name the
        # zone-set with that exact folder product so the scan finds it. Otherwise
        # keep a selected product, else fall back to the filename suggestion.
        prod = product_from_raw_path(path)
        if prod:
            self.var_product.set(prod)
        elif not self.var_product.get().strip():
            self.var_product.set(suggest_product(path))    # editable suggestion
        self._refresh_versions(self.var_product.get())
        self.boxes = list(boxes_for(self.var_product.get(), self.version))
        self._poly = []
        self.redraw()
        return True

    def open_raw(self):
        path = filedialog.askopenfilename(
            title="Reference RAW", initialdir=self._raw_initialdir(),
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")])
        if path:
            self.load_image(path)

    def _raw_initialdir(self):
        """Start the file dialog inside the selected product's newest Image
        Activescale folder, so opened RAWs carry the folder-product identity."""
        root = self._root_getter() if self._root_getter else ""
        prod = self.var_product.get().strip()
        if root and os.path.isdir(root) and prod:
            days = pic_days(root, prod, False)
            if days:
                leaf = pic_leaf(root, "Image", prod, days[0])
                if os.path.isdir(leaf):
                    return leaf
        return os.path.dirname(self.raw_path) if self.raw_path else ""

    def load_for(self, product, raw_path):
        """From the Scan tab: draw the missing product on this RAW."""
        if self.load_image(raw_path):
            self.var_product.set(product)
            self._refresh_versions(product)
            self.boxes = list(boxes_for(product, self.version))
            self._poly = []
            self.redraw()

    def _switch_version(self):
        self.version = self.var_version.get()
        self.boxes = list(boxes_for(self.var_product.get().strip(), self.version))
        self.selected = None
        self._poly = []
        self.redraw()

    def _new_version(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("New version", "Version name:", parent=self)
        if not name:
            return
        vals = list(self.cmb_version["values"])
        if name not in vals:
            vals.append(name)
        self.cmb_version["values"] = vals
        self.version = name
        self.var_version.set(name)
        self.boxes, self.selected = [], None
        self.redraw()

    def _delete_version(self):
        prod, ver = self.var_product.get().strip(), self.version
        if not prod or not ver:
            return
        if ver not in versions_of(prod):
            self._refresh_versions(prod)
            self._switch_version()
            return
        if messagebox.askyesno("Delete version", f"Delete '{ver}' of {prod}?"):
            delete_version(prod, ver)
            self._refresh_versions(prod)
            self._switch_version()

    def save(self):
        prod, ver = self.var_product.get().strip(), self.version or "v1"
        if not self.raw_path or not prod:
            messagebox.showerror("Nothing to save", "Open a RAW and name a product.")
            return
        save_version(prod, ver, self.boxes, os.path.basename(self.raw_path))
        self._refresh_products()
        self._refresh_versions(prod, select=ver)
        self.var_status.set(f"Saved {len(self.boxes)} box(es) -> {prod} / {ver} (active)")

    def clear(self):
        self.boxes, self.selected = [], None
        self._poly, self.boxes_preview, self._sel_vertex = [], None, None
        self.redraw()

    def _key_active(self):
        """Keyboard actions apply only when the editor's tab is actually shown --
        stops ROI keys firing while the Scan tab is up (shared toplevel bindings)."""
        return bool(self.canvas.winfo_ismapped())

    def delete_selected(self, _evt=None):
        """Delete key: remove the picked polygon vertex if one is selected (and the
        polygon keeps >= 3 points); otherwise delete the whole selected shape."""
        if not self._key_active():
            return
        if self._delete_vertex():
            return
        if self.selected is not None and self.selected < len(self.boxes):
            del self.boxes[self.selected]
            self.selected = None
            self._sel_vertex = None
            self.redraw()

    def _delete_vertex(self):
        """Remove the selected polygon vertex. Returns True if it handled the key."""
        if self.selected is None or self.selected >= len(self.boxes) \
                or self._sel_vertex is None:
            return False
        sh = self.boxes[self.selected]
        if sh["type"] != "poly" or self._sel_vertex >= len(sh["pts"]):
            return False
        if len(sh["pts"]) <= 3:                        # a polygon needs >= 3 vertices
            self.var_status.set("Polygon needs at least 3 points -- delete the shape instead.")
            return True
        pts = [p for j, p in enumerate(sh["pts"]) if j != self._sel_vertex]
        self.boxes[self.selected] = {"type": "poly", "pts": pts}
        self._sel_vertex = None
        self.redraw()
        return True

    def nudge_selected(self, dx, dy, mult=1):
        """Arrow keys move the selected shape by dx,dy unit dir * step px
        (Shift = x10). Step read from toolbar spinbox. Select tool only."""
        if not self._key_active():
            return
        if self.selected is None or self.selected >= len(self.boxes):
            return
        focus = self.focus_get()
        if isinstance(focus, (ttk.Entry, tk.Entry, ttk.Combobox, tk.Spinbox, ttk.Spinbox)):
            return                                     # typing in a field, not nudging
        try:
            step = max(1, int(self.var_step.get()))
        except (tk.TclError, ValueError):
            step = 1
        step *= mult
        moved = shape_move(self.boxes[self.selected], dx * step, dy * step)
        self.boxes[self.selected] = self.clamp_shape(moved)
        self.redraw()

    # ---- polygon build ----------------------------------------------------
    def _finish_poly(self):
        """Close the in-progress polygon (needs >= 3 vertices) into a shape."""
        if not self._key_active():
            return
        if len(self._poly) >= 3:
            sh = self.clamp_shape({"type": "poly", "pts": [list(p) for p in self._poly]})
            self.boxes.append(sh)
            self.selected = len(self.boxes) - 1
        self._poly = []
        self.redraw()

    def _cancel_poly(self):
        if not self._key_active():
            return
        if self._poly:
            self._poly = []
            self.redraw()

    def _undo_poly_point(self):
        """Drop the last-placed polygon vertex (step back a wrong click)."""
        if not self._key_active():
            return
        if self._poly:
            self._poly.pop()
            self.redraw()

    def _backspace(self, evt=None):
        """While drawing a polygon Backspace undoes the last point; otherwise it
        deletes the selected shape."""
        if self._poly:
            self._undo_poly_point()
        else:
            self.delete_selected(evt)

    # ---- mouse ------------------------------------------------------------
    def on_press(self, evt):
        self.canvas.focus_set()          # steal focus off spinbox so arrows nudge boxes
        if self.img is None:
            return
        tf = self._tf()
        nx, ny = self.c2n(evt.x, evt.y, tf)
        if self.tool_draw():
            shape = self.var_shape.get()
            if shape == "poly":
                if len(self._poly) >= 3:               # click near start vertex = close
                    fx, fy = self._poly[0]
                    if ((nx - fx) ** 2 + (ny - fy) ** 2) ** 0.5 * tf[0] <= 2 * CLICK_TOL:
                        self._finish_poly()
                        return
                self._poly.append([nx, ny])            # otherwise add a vertex
            elif shape == "rect":
                self._drag = ("draw_rect", nx, ny)     # drag out a rectangle
            elif shape == "circle":
                self._drag = ("draw_circle", nx, ny)
        else:
            hid = self.handle_at(nx, ny, tf)
            if hid is not None:
                self._drag = ("resize", hid)           # grab a resize grip / vertex
                self._sel_vertex = hid if isinstance(hid, int) else None
            elif self._insert_vertex_at(nx, ny, tf):   # click on selected poly's edge = new point
                self._drag = ("resize", self._sel_vertex)   # drag it straight away
            else:
                hit = self.box_at(nx, ny)
                self._sel_vertex = None
                if hit is not None:
                    self.selected, self._drag = hit, ("move", nx, ny)
                else:
                    self.selected, self._drag = None, ("pan", evt.x, evt.y)
        self.redraw()

    def on_double(self, _evt):
        if self.tool_draw() and self.var_shape.get() == "poly":
            if self._poly:
                self._poly.pop()          # drop the duplicate vertex the 2nd click added
            self._finish_poly()

    def _insert_vertex_at(self, nx, ny, tf):
        """If (nx,ny) lands on an edge of the *selected* polygon, insert a vertex
        there and select it. Returns True if a vertex was added."""
        if self.selected is None or self.selected >= len(self.boxes):
            return False
        sh = self.boxes[self.selected]
        if sh["type"] != "poly":
            return False
        pts = sh["pts"]
        best_i, best_d = None, None
        for i in range(len(pts)):                       # each edge i -> i+1 (wraps)
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % len(pts)]
            d = _pt_seg_dist(nx, ny, ax, ay, bx, by)
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        if best_d is None or best_d * tf[0] > 2 * CLICK_TOL:
            return False
        self.boxes[self.selected] = {"type": "poly",
                                     "pts": pts[:best_i + 1] + [[nx, ny]] + pts[best_i + 1:]}
        self._sel_vertex = best_i + 1
        return True

    def on_right(self, _evt):
        if self._poly:                    # right-click closes the polygon (no keyboard needed)
            self._finish_poly()

    def _mid_pan_start(self, evt):
        self._drag = ("pan", evt.x, evt.y)

    def on_drag(self, evt):
        if not self._drag:
            return
        tf = self._tf()
        nx, ny = self.c2n(evt.x, evt.y, tf)
        mode = self._drag[0]
        if mode == "pan":
            _, lx, ly = self._drag
            self.pan[0] += evt.x - lx
            self.pan[1] += evt.y - ly
            self._drag = ("pan", evt.x, evt.y)
        elif mode == "move" and self.selected is not None:
            _, lx, ly = self._drag                     # move by cursor delta in native px
            moved = shape_move(self.boxes[self.selected], nx - lx, ny - ly)
            self.boxes[self.selected] = self.clamp_shape(moved)
            self._drag = ("move", nx, ny)
        elif mode == "resize" and self.selected is not None:
            self.boxes[self.selected] = shape_resize(self.boxes[self.selected],
                                                     self._drag[1], nx, ny)
        elif mode == "draw_rect":
            _, ax, ay = self._drag
            self.boxes_preview = {"type": "rect", "cx": (ax + nx) / 2, "cy": (ay + ny) / 2,
                                  "w": max(MIN_SIDE, abs(nx - ax)), "h": max(MIN_SIDE, abs(ny - ay))}
        elif mode == "draw_circle":
            _, ax, ay = self._drag
            r = max(MIN_SIDE, ((nx - ax) ** 2 + (ny - ay) ** 2) ** 0.5)
            self.boxes_preview = {"type": "circle", "cx": ax, "cy": ay, "r": r}
        self.redraw()

    def on_release(self, _evt):
        drag = self._drag
        preview = self.boxes_preview
        s = self._tf()[0]                                          # native->canvas scale
        if drag and drag[0] == "draw_rect" and preview is not None \
                and max(preview["w"], preview["h"]) * s >= CLICK_TOL:
            self.boxes.append(self.clamp_shape(preview))           # dragged rectangle
            self.selected = len(self.boxes) - 1
        elif drag and drag[0] == "draw_circle" and preview is not None \
                and preview["r"] * s >= CLICK_TOL:
            self.boxes.append(self.clamp_shape(preview))
            self.selected = len(self.boxes) - 1
        self.boxes_preview = None
        self._drag = None
        self.redraw()

    def on_motion(self, _evt):
        pass                             # live rect/circle is drawn during drag, not hover

    # ---- zoom / pan -------------------------------------------------------
    def _wheel_pan(self, evt, horizontal):
        step = (evt.delta / 120) * 60
        self.pan[0 if horizontal else 1] += step
        self.redraw()

    def _wheel_zoom(self, evt):
        if self.img is None:
            return
        nx, ny = self.c2n(evt.x, evt.y, self._tf())
        self.zoom = min(max(self.zoom * (1.2 if evt.delta > 0 else 1 / 1.2), 0.1), 20)
        cx, cy = self.n2c(nx, ny, self._tf())
        self.pan[0] += evt.x - cx
        self.pan[1] += evt.y - cy
        self.redraw()

    def _zoom_step(self, f):
        self.zoom = min(max(self.zoom * f, 0.1), 20)
        self.redraw()

    def _fit(self):
        self.zoom, self.pan = 1.0, [0.0, 0.0]
        self.redraw()

    # ---- draw -------------------------------------------------------------
    def redraw(self):
        c = self.canvas
        c.delete("all")
        if self.img is None:
            return
        tf = self._tf()
        s, x0, y0, cw, ch = tf
        vx0, vy0 = max(0, int(-x0 / s)), max(0, int(-y0 / s))
        vx1 = min(self.img_w, int((cw - x0) / s) + 1)
        vy1 = min(self.img_h, int((ch - y0) / s) + 1)
        if vx1 > vx0 and vy1 > vy0:
            sw, sh = max(1, int((vx1 - vx0) * s)), max(1, int((vy1 - vy0) * s))
            interp = cv2.INTER_NEAREST if s > 1 else cv2.INTER_AREA
            sub = cv2.resize(self.img[vy0:vy1, vx0:vx1], (sw, sh), interpolation=interp)
            self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)))
            c.create_image(x0 + vx0 * s, y0 + vy0 * s, anchor="nw", image=self.photo)
        for i, sh in enumerate(self.boxes):
            sel = (i == self.selected)
            self._draw_shape(c, sh, tf, "#ff3b3b" if sel else "#00c8ff", 2 if sel else 1)
            bx0, by0, _, _ = shape_bbox(sh)
            lx, ly = self.n2c(bx0, by0, tf)
            c.create_text(lx + 8, ly - 8, text=str(i), fill="#00c8ff")
            if sel:                                    # resize grips on the selected shape
                for hid, hx, hy in shape_handles(sh):
                    gx, gy = self.n2c(hx, hy, tf)
                    picked = (hid == self._sel_vertex)  # highlight the chosen vertex
                    fill = "#00ff88" if picked else "#ff3b3b"
                    r = 6 if picked else 4
                    c.create_rectangle(gx - r, gy - r, gx + r, gy + r,
                                       fill=fill, outline="#ffffff")
        if self.boxes_preview is not None:             # live rect/circle being dragged
            self._draw_shape(c, self.boxes_preview, tf, "#ffffff", 1, dash=(4, 3))
        if self._poly:                                 # in-progress polygon
            pts = [self.n2c(x, y, tf) for x, y in self._poly]
            if len(pts) >= 2:
                c.create_line(*sum(([px, py] for px, py in pts), []),
                              fill="#ffff66", width=1)
            for px, py in pts:
                c.create_rectangle(px - 3, py - 3, px + 3, py + 3, fill="#ffff66", outline="")
            fx, fy = pts[0]                             # first vertex = the close target
            ring = "#00ff88" if len(pts) >= 3 else "#ffff66"
            c.create_oval(fx - 6, fy - 6, fx + 6, fy + 6, outline=ring, width=2)
        shape = self.var_shape.get()
        if self.tool_draw() and shape == "poly":
            hint = ("  |  polygon: click=add, Backspace/Ctrl+Z=undo point, "
                    "click green start / right-click / double-click / Enter = close, Esc = cancel")
        elif self.tool_draw() and shape in ("rect", "circle"):
            hint = f"  |  {shape}: click-drag to draw"
        elif not self.tool_draw() and self._sel_vertex is not None:
            hint = "  |  vertex: drag=move, Del/Backspace=delete point"
        elif not self.tool_draw():
            hint = "  |  click shape=select, click its line=add point, drag=move/resize, Del=remove"
        else:
            hint = "  |  wheel=pan  Ctrl+wheel=zoom  Del=remove"
        self.var_status.set(
            f"{os.path.basename(self.raw_path) or '(no image)'}  |  "
            f"{self.img_w}x{self.img_h}px  zoom {self.zoom:.2f}  |  "
            f"{self.var_product.get() or '(no product)'} / {self.version}  |  "
            f"{len(self.boxes)} shape(s)  |  "
            f"{'draw ' + self.var_shape.get() if self.tool_draw() else 'select/move/resize'}"
            f"{hint}")

    def _draw_shape(self, c, sh, tf, colour, width, dash=None):
        opt = {"outline": colour, "width": width}
        if dash:
            opt["dash"] = dash
        t = sh["type"]
        if t == "circle":
            x0, y0 = self.n2c(sh["cx"] - sh["r"], sh["cy"] - sh["r"], tf)
            x1, y1 = self.n2c(sh["cx"] + sh["r"], sh["cy"] + sh["r"], tf)
            c.create_oval(x0, y0, x1, y1, **opt)
        elif t == "poly":
            flat = sum(([*self.n2c(x, y, tf)] for x, y in sh["pts"]), [])
            c.create_polygon(*flat, fill="", **opt)
        else:
            bx0, by0, bx1, by1 = shape_bbox(sh)
            x0, y0 = self.n2c(bx0, by0, tf)
            x1, y1 = self.n2c(bx1, by1, tf)
            c.create_rectangle(x0, y0, x1, y1, **opt)


# ----------------------------------------------------------------- scanning --



# ==================================================================== entry --

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
