"""
Re-judge AVI NG calls: crop -> predict -> measure -> suggested verdict; a human
confirms every unit (docs/adr/0002). Verdicts come from measured mask size, not
detection presence (docs/adr/0001).

Inputs are two independently picked folders (they need not share a parent):
  - Fail folder:  the machine's RESULT images (annotated, bounding boxes)
  - Image folder: RAW images. A Fail image's RAW is matched by serial -- the
    first _-separated token -- preferring the exact stem twin (name minus
    _Fail), then same station+view, taking the RAW closest AT-OR-BEFORE the
    Fail's time (a later retest RAW may show the unit repositioned).

After Start the watcher polls the Fail folder (rejudge: poll_seconds) and
processes every new image automatically, once its size stops changing (copy
finished) and its RAW exists. A missing RAW errors the unit after RAW_WAIT_S
but keeps retrying for the session; a late RAW replaces the error row (the
human verdict, if any, is kept). Fail images already recorded in their date's
units.csv are skipped on restart.

Flow per Fail image:
  1. Recover defect boxes from the RESULT image (crop_ng_regions.find_defect_boxes).
  2. Cut a CROP_SIZE window from the RAW image around each box and run the
     selected YOLO-seg weight on it.
  3. If a predicted mask touches the crop border (defect likely continues
     outside), re-crop centred on the mask and re-predict, growing the window
     until the whole defect fits ("truncated mask" rule).
  4. Measure each mask -- scratch: longest dimension (mm); scuffmark: area (mm²)
     -- and gate against the size criteria in config.yaml [rejudge:].
  5. Suggest a verdict per crop and per unit (worst-crop ladder
     NG > Review > Cleaning > Overkill, then unit count / total-area limits).
  6. The user reviews overlay + crops + table and clicks the final verdict.

Outputs are grouped by the DATE token in the filename ({serial}_{datetime}_...):
    rejudge_output/{YYYYMMDD}/units.csv    suggestion AND human verdict per inspection
    rejudge_output/{YYYYMMDD}/crops.csv    one row per detection
    rejudge_output/{YYYYMMDD}/serials.csv  unit disposition: worst verdict per serial
    rejudge_output/{YYYYMMDD}/images/{serial}_{station-view}/...
Rewrites merge with the CSV already on disk, so earlier sessions' rows survive
(latest state per inspection; verdict corrections overwrite -- docs/adr/0003).

Run:            python rejudge_gui.py
Self-test only: python rejudge_gui.py --selftest
"""

import csv
import glob
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

from crop_ng_regions import find_defect_boxes, CROP_SIZE, IMAGE_EXT, FAIL_STATUS


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

    def load(self):
        from ultralytics import YOLO
        self.model = YOLO(self.weights_path)

    def predict(self, img):
        """[(class_name, conf, poly_local Nx2)] for one BGR crop."""
        r = self.model.predict(img, conf=self.settings.conf_threshold,
                               imgsz=640, verbose=False)[0]
        out = []
        if r.masks is None:
            return out
        for poly, cls, conf in zip(r.masks.xy, r.boxes.cls, r.boxes.conf):
            if len(poly) >= 3:
                name = self.model.names[int(cls)]
                out.append((str(name), float(conf), np.asarray(poly, np.float32)))
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


def render_overlay(unit: UnitResult, annotate=True):
    """Downscaled RAW; with annotate, crop windows (blue) and per-detection
    masks colored by verdict -- where on the unit each defect sits."""
    img = unit.overlay_base.copy()
    if not annotate:
        return img
    s = unit.overlay_scale
    for crop in unit.crops:
        x0, y0, x1, y1 = [int(v * s) for v in crop.window]
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
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
        self.q = queue.Queue()
        self.worker = None
        self.stop_evt = None
        self._photo = {}             # keep PhotoImage refs alive

        self._build_toolbar()
        self._build_body()
        root.bind("<Key>", self._on_key)
        root.after(100, self._poll)

    # ---- layout ----
    def _build_toolbar(self):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.root, padding=(4, 4, 4, 0))
        top.pack(fill="x")
        ttk.Label(top, text="Fail folder (RESULT):").pack(side="left")
        self.var_fail = tk.StringVar()
        ttk.Entry(top, textvariable=self.var_fail, width=44).pack(side="left", padx=2)
        ttk.Button(top, text="Browse…", command=lambda: self._pick_dir(
            self.var_fail, "Fail folder - RESULT images with bounding boxes")).pack(side="left")
        ttk.Label(top, text="  Image folder (RAW):").pack(side="left")
        self.var_image = tk.StringVar()
        ttk.Entry(top, textvariable=self.var_image, width=44).pack(side="left", padx=2)
        ttk.Button(top, text="Browse…", command=lambda: self._pick_dir(
            self.var_image, "Image folder - RAW images (matched by serial)")).pack(side="left")

        bar = ttk.Frame(self.root, padding=4)
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
        self.var_status = tk.StringVar(value="Pick the Fail folder, the Image folder and a weight, then Start.")
        ttk.Label(bar, textvariable=self.var_status).pack(side="left", padx=10)

    def _build_body(self):
        tk, ttk = self.tk, self.ttk
        pane = ttk.PanedWindow(self.root, orient="horizontal")
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
        for cv in (self.cv_overlay, self.cv_crop):
            cv.bind("<Configure>", lambda e: self._redraw())

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
        ttk.Label(vb, text="   (keys 1-4; ←/→ = prev/next unit)").pack(side="left")

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
        fail_dir = self.var_fail.get().strip()
        raw_dir = self.var_image.get().strip()
        if not (os.path.isdir(fail_dir) and os.path.isdir(raw_dir)):
            messagebox.showerror("No input",
                "Pick the Fail folder (RESULT images) and the Image folder (RAW images).")
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
        already = self._already_recorded(fail_dir)
        self.btn_start["text"] = "Stop"
        self.var_status.set("Loading model…")
        engine = Engine(self.weights[self.var_weight.get()], self.settings)
        self.stop_evt = threading.Event()
        self.worker = threading.Thread(
            target=self._watch, args=(engine, fail_dir, raw_dir, already, self.stop_evt),
            daemon=True)
        self.worker.start()

    def _already_recorded(self, fail_dir):
        """Basenames of Fail images already in their date's units.csv (earlier
        sessions). Read on the GUI thread, before the watcher can write."""
        already, per_date = set(), {}
        for p in glob.glob(os.path.join(fail_dir, f"*{IMAGE_EXT}")):
            _, date, _, _ = parse_fail_name(p)
            if date not in per_date:
                per_date[date] = self.writer.recorded(date)
            if os.path.basename(p) in per_date[date]:
                already.add(os.path.basename(p))
        return already

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

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "unit":
                    unit = msg[1]
                    idx = next((i for i, u in enumerate(self.units)
                                if os.path.basename(u.result_path)
                                == os.path.basename(unit.result_path)), None)
                    if idx is None:
                        self.units.append(unit)
                        self._add_row(unit)
                    else:                       # re-run of a shown unit (late RAW)
                        unit.human, unit.human_time = \
                            self.units[idx].human, self.units[idx].human_time
                        self.units[idx] = unit
                        self._refresh_row(idx)
                        if self.current == idx:
                            self._show_unit(idx)
                    if self._save_results(unit):
                        self.var_status.set(f"Processed {len(self.units)}: "
                                            f"{unit.serial} [{unit.suggested}]")
                    if self.current is None:
                        self._show_unit(0)
                elif msg[0] == "done":
                    self.btn_start["text"] = "Start"
                    self.btn_start["state"] = "normal"
                    self.var_status.set(f"Stopped - {len(self.units)} unit(s) this session. "
                                        f"Output: {self.writer.root}")
                elif msg[0] == "fatal":
                    self.btn_start["text"] = "Start"
                    self.btn_start["state"] = "normal"
                    self.var_status.set(msg[1])
                elif msg[0] == "status":
                    self.var_status.set(msg[1])
        except queue.Empty:
            pass
        finally:
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
        if not (0 <= idx < len(self.units)):
            return
        self.current, self.current_crop = idx, 0
        u = self.units[idx]
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
            self._redraw()

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
        from PIL import Image, ImageTk
        cw = max(50, canvas.winfo_width())
        ch = max(50, canvas.winfo_height())
        h, w = bgr.shape[:2]
        s = min(cw / w, ch / h)
        img = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))))
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self._photo[key] = photo
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo)

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


# ---------------------------------------------------------------- selftest --

def selftest():
    """Exercise the pure logic (no model, no GUI)."""
    ok = [0]

    def check(name, cond):
        ok[0] += not cond
        print(("PASS " if cond else "FAIL ") + name)

    check("window centred", crop_window(4000, 3000, 2000, 1500, 640) == (1680, 1180, 2320, 1820))
    check("window shifted at corner", crop_window(4000, 3000, 10, 10, 640) == (0, 0, 640, 640))
    check("window on small image", crop_window(500, 400, 250, 200, 640) == (0, 0, 500, 400))

    sq = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.float32)
    ln, ar = measure_poly(sq)
    check("square measure", abs(ln - 100) < 1 and abs(ar - 10000) < 1)

    win = (90, 90, 730, 730)
    check("touch detected", touches_border(sq - 8, win, 4000, 3000))
    check("no touch inside", not touches_border(sq + 100, win, 4000, 3000))
    check("raw edge is not truncation", not touches_border(
        np.array([[1, 1], [50, 1], [50, 50], [1, 50]], np.float32), (0, 0, 640, 640), 4000, 3000))

    s = Settings(scratch_length_mm=2.0, scuffmark_area_mm2=1.0,
                 default_mm_per_px=0.01, max_defect_count=3, max_total_area_mm2=5.0)

    def det(name, poly, conf=0.9):
        d = Detection(name, conf, poly)
        d.length_px, d.area_px = measure_poly(d.poly)
        return d

    long_scratch = np.array([[0, 0], [300, 0], [300, 4], [0, 4]], np.float32)   # 3mm
    short_scratch = np.array([[0, 0], [50, 0], [50, 4], [0, 4]], np.float32)    # 0.5mm
    near_scratch = np.array([[0, 0], [190, 0], [190, 4], [0, 4]], np.float32)   # 1.9mm

    u = UnitResult("S", "d", "CM1", "2", "", "")
    u.crops = [CropResult(0, "tight", (0, 0, 640, 640),
                          [det("scratch", long_scratch), det("contam_dust", sq)]),
               CropResult(1, "tight", (0, 0, 640, 640), [det("scratch", short_scratch)]),
               CropResult(2, "zone", (0, 0, 640, 640), [])]
    judge_unit(u, s)
    check("oversize scratch -> NG", u.crops[0].detections[0].verdict == NG)
    check("contamination -> Cleaning", u.crops[0].detections[1].verdict == CLEANING)
    check("undersize scratch -> Overkill", u.crops[1].detections[0].verdict == OVERKILL)
    check("empty crop -> Review", u.crops[2].verdict == REVIEW)
    check("ladder picks NG", u.suggested == NG)

    u2 = UnitResult("S2", "d", "CM1", "2", "", "")
    u2.crops = [CropResult(0, "tight", (0, 0, 640, 640), [det("scratch", near_scratch)])]
    judge_unit(u2, s)
    d0 = u2.crops[0].detections[0]
    check("near-limit flagged", d0.verdict == OVERKILL and d0.near_limit)

    u3 = UnitResult("S3", "d", "CM1", "2", "", "")
    u3.crops = [CropResult(i, "tight", (0, 0, 640, 640),
                           [det("spot", sq + 300 * i)]) for i in range(4)]
    judge_unit(u3, s)
    check("count limit -> unit NG", u3.suggested == NG and "count" in u3.unit_note)

    u4 = UnitResult("S4", "d", "CM1", "2", "", "")
    u4.crops = [CropResult(0, "tight", (0, 0, 640, 640),
                           [det("scratch", short_scratch), det("scratch", short_scratch, 0.8)])]
    judge_unit(u4, s)
    check("duplicate detections deduped", u4.defect_count == 1)

    # overlap inflation: one physical mark, counted once, area union not sum
    band = np.array([[0, 0], [100, 0], [100, 40], [0, 40]], np.float32)
    u7 = UnitResult("S7", "d", "CM1", "2", "", "")
    u7.crops = [CropResult(0, "tight", (0, 0, 640, 640), [det("scratch", band)]),
                CropResult(1, "tight", (0, 0, 640, 640),
                           [det("scuffmark", band + [60, 0])])]   # 40% shared, bbox IoU 0.25
    judge_unit(u7, s)
    check("partial cross-class overlap counts once", u7.defect_count == 1)
    check("total area is union not sum", 6400 <= u7.total_area_px <= 6800)

    u8 = UnitResult("S8", "d", "CM1", "2", "", "")
    u8.crops = [CropResult(0, "tight", (0, 0, 640, 640),
                           [det("scratch", sq), det("spot", sq / 2 + 75)])]
    judge_unit(u8, s)
    check("contained mask counts once", u8.defect_count == 1)

    u9 = UnitResult("S9", "d", "CM1", "2", "", "")
    u9.crops = [CropResult(0, "tight", (0, 0, 640, 640),
                           [det("scratch", band), det("scratch", band + 400)])]
    judge_unit(u9, s)
    check("separate defects still count twice",
          u9.defect_count == 2 and 8200 <= u9.total_area_px <= 8600)

    u10 = UnitResult("S10", "d", "CM1", "2", "", "")
    u10.crops = [CropResult(0, "tight", (0, 0, 640, 640),
                            [det("contam_stain", sq), det("scratch", short_scratch)])]
    judge_unit(u10, s)
    check("contamination excluded from unit limits",
          u10.defect_count == 1 and u10.total_area_px < 1000)

    no_cal = Settings(scratch_length_mm=2.0)         # calibration missing
    u5 = UnitResult("S5", "d", "CM1", "2", "", "")
    u5.crops = [CropResult(0, "tight", (0, 0, 640, 640), [det("scratch", long_scratch)])]
    judge_unit(u5, no_cal)
    check("no calibration -> Review", u5.crops[0].detections[0].verdict == REVIEW)

    u6 = UnitResult("S6", "d", "CM9", "9", "", "")
    u6.crops = [CropResult(0, "tight", (0, 0, 640, 640), [det("Blister", sq)])]
    judge_unit(u6, s)
    check("unknown class -> Review", u6.crops[0].detections[0].verdict == REVIEW)

    sv = Settings(scratch_length_mm=2.0, default_mm_per_px=0.5,
                  per_station_view={"CM1-2": 0.01})
    check("per-station-view calibration wins", sv.mm_per_px("CM1-2") == 0.01
          and sv.mm_per_px("CM9-9") == 0.5)

    # _refine truncation bookkeeping, with predict stubbed (no model needed).
    # edge_poly hugs the right border of its original (0,0,640,640) crop window.
    raw = np.zeros((3000, 4000, 3), np.uint8)
    edge_poly = np.array([[600, 300], [639, 300], [639, 320], [600, 320]], np.float32)
    eng = Engine("", s)

    eng.predict = lambda img: []                       # defect never re-found
    d1 = det("scratch", edge_poly.copy())
    eng._refine(raw, d1)
    check("unrefindable mask stays truncated", d1.truncated)

    win0 = crop_window(4000, 3000, *edge_poly.mean(axis=0), CROP_SIZE)
    grown = np.array([[590, 295], [660, 295], [660, 325], [590, 325]], np.float32)
    eng.predict = lambda img: [("scratch", 0.95, grown - win0[:2])]  # clears border
    d2 = det("scratch", edge_poly.copy())
    eng._refine(raw, d2)
    check("re-fit mask clears truncation", not d2.truncated and d2.conf == 0.95)

    def predict_hug(img):                              # hugs every window's border
        h, w = img.shape[:2]
        return [("scratch", 0.9, np.array([[250, 290], [w - 1, 290],
                                           [w - 1, 330], [250, 330]], np.float32))]
    eng.predict = predict_hug
    d3 = det("scratch", edge_poly.copy())
    eng._refine(raw, d3)
    check("mask cut off at max window stays truncated", d3.truncated)

    # folder watching / matching / per-date output (real files in a temp dir)
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp()
    try:
        full = os.path.join(tmp, "ZQ1_20260707103000_CM1_2_x_Fail.jpg")
        short = os.path.join(tmp, "ZQ2-noparts.jpg")
        for p in (full, short):
            open(p, "w").close()
        check("parse full fail name",
              parse_fail_name(full) == ("ZQ1", "20260707", "CM1", "2"))
        serial, date, station, view = parse_fail_name(short)
        check("parse short name falls back",
              serial == "ZQ2-noparts" and len(date) == 8 and date.isdigit()
              and (station, view) == ("?", "?"))

        raw_dir = os.path.join(tmp, "img")
        os.makedirs(raw_dir)
        twin = os.path.join(raw_dir, "ZQ1_20260707103000_CM1_2_x" + IMAGE_EXT)
        early = os.path.join(raw_dir, "ZQ1_20260707090000_CM1_2" + IMAGE_EXT)
        retest = os.path.join(raw_dir, "ZQ1_20260707110000_CM1_2" + IMAGE_EXT)
        other_sv = os.path.join(raw_dir, "ZQ1_20260707100000_CM9_9" + IMAGE_EXT)
        skew = os.path.join(raw_dir, "ZQ3_20260707103001_CM1_2" + IMAGE_EXT)
        for p in (twin, early, retest, other_sv, skew):
            open(p, "w").close()
        check("raw match prefers exact twin",
              find_raw_for("ZQ1", "CM1", "2", raw_dir,
                           fail_stem="ZQ1_20260707103000_CM1_2_x_Fail") == twin)
        # fail at 10:59 -- retest RAW at 11:00 is nearer in time but AFTER, so
        # the 10:30 at-or-before RAW must win
        check("raw match: at-or-before beats nearer later retest",
              find_raw_for("ZQ1", "CM1", "2", raw_dir,
                           fail_stem="ZQ1_20260707105900_CM1_2_Fail") == twin)
        check("raw match: same view beats closer other view",
              find_raw_for("ZQ1", "CM1", "2", raw_dir,
                           fail_stem="ZQ1_20260707095900_CM1_2_Fail") == early)
        check("raw match: clock-skew falls back to closest after",
              find_raw_for("ZQ3", "CM1", "2", raw_dir,
                           fail_stem="ZQ3_20260707103000_CM1_2_Fail") == skew)
        check("no raw -> None", find_raw_for("NOPE", "CM1", "2", raw_dir) is None)

        out = os.path.join(tmp, "out")
        ua = UnitResult("A", "20260707", "CM1", "2", "", os.path.join(tmp, "A_x_Fail.jpg"))
        ub = UnitResult("B", "20260707", "CM1", "2", "", os.path.join(tmp, "B_x_Fail.jpg"))
        DateWriter(out).write_all([ua])                # session 1
        w2 = DateWriter(out)                           # session 2 merges, not clobbers
        w2.write_all([ub])
        check("date CSV merges sessions",
              w2.recorded("20260707") == {"A_x_Fail.jpg", "B_x_Fail.jpg"})
        ua.human = "NG"
        w2.write_all([ua, ub])                         # same units replace their rows
        with open(os.path.join(out, "20260707", "units.csv"),
                  newline="", encoding="utf-8-sig") as f:
            rows = {r["serial"]: r for r in csv.DictReader(f)}
        check("same unit row replaced not duplicated",
              len(rows) == 2 and rows["A"]["human_verdict"] == "NG")

        # unit disposition: serial A gains a second, unjudged inspection
        ua2 = UnitResult("A", "20260707", "CM3", "1", "",
                         os.path.join(tmp, "A_y_Fail.jpg"))
        w2.write_all([ua, ua2, ub])
        with open(os.path.join(out, "20260707", "serials.csv"),
                  newline="", encoding="utf-8-sig") as f:
            srows = {r["serial"]: r for r in csv.DictReader(f)}
        check("serial rollup: worst across inspections, incomplete",
              srows["A"]["worst_human"] == "NG" and srows["A"]["inspections"] == "2"
              and srows["A"]["judged"] == "1" and srows["A"]["complete"] == "False")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(("ALL OK" if ok[0] == 0 else f"{ok[0]} FAILURES"))
    return ok[0]


def main():
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    import tkinter as tk
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
