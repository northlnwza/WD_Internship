"""
Surface scan (ROI) - prototype, single window.

Model as a full-surface second net: instead of cropping regions the machine
already flagged (rejudge_gui.py), crop the fixed 640 "master boxes" a user drew
for a PRODUCT and run the model on each. Two tabs, one file:

  Template : open a reference RAW; pan (drag / two-finger scroll) and zoom
             (Ctrl+scroll or the zoom buttons) the view. Pick a tool -- Draw
             drops 640 boxes at the cursor (shown at true scale), Select moves
             boxes and pans empty space. Boxes are saved under a PRODUCT name you
             choose (a filename-based suggestion is prefilled but fully editable)
             and each product can hold several named VERSIONS of a box layout.
  Scan     : pick ONE product (and version) and a RAW folder; every RAW in the
             folder is cropped with that product's boxes, predicted, and judged
             with the SAME rules + calibration as rejudge_gui, then written to
             rejudge_output/... Results are shown like rejudge_gui -- units list,
             full-image overlay with masks coloured by verdict, per-box crops,
             and detections table. The filename is parsed only to file the output
             (serial/date/station/view), never to choose the boxes.

ROIs are stored as shapes in native RAW px in rois.json, next to config.yaml,
keyed by product name. A shape is one of rect / circle / poly; the model always
crops the shape's bounding box and zeroes everything outside the shape. Legacy
bare [cx,cy] boxes auto-migrate to BOX_SIZE squares on load:

    { "WidgetA-topcam": { "box_size": 640, "active": "v1", "versions": { "v1": {
        "reference_raw": "...", "boxes": [
            {"type": "rect",   "cx": 0, "cy": 0, "w": 640, "h": 640},
            {"type": "circle", "cx": 0, "cy": 0, "r": 300},
            {"type": "poly",   "pts": [[x,y], ...]} ] } } } }

Usage:  python surface_scan.py
"""

import glob
import json
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from rejudge_gui import (
    OVERLAY_MAX_W, REVIEW, IMAGE_EXT, CropResult, Detection, DateWriter,
    Engine, Settings, UnitResult, app_dir, find_weights, judge_unit,
    measure_poly, parse_fail_name, render_crop, render_overlay,
)

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
    for i, shape in enumerate(boxes):
        bx0, by0, bx1, by1 = shape_bbox(shape)
        x0 = max(0, int(round(bx0)))
        y0 = max(0, int(round(by0)))
        x1 = min(raw_w, int(round(bx1)))
        y1 = min(raw_h, int(round(by1)))
        if x1 <= x0 or y1 <= y0:
            continue                                   # shape fully off-image
        win = (x0, y0, x1, y1)
        sub = raw[y0:y1, x0:x1]
        if shape["type"] != "rect":                    # zero everything outside the region
            mask = shape_mask(shape, x0, y0, x1 - x0, y1 - y0)
            sub = cv2.bitwise_and(sub, sub, mask=mask)
        crop = CropResult(i, "roi", win, shape=shape)
        for name, conf, local in engine.predict(sub):
            det = Detection(name, conf, local + np.array([x0, y0], np.float32))
            clipped = clip_poly_to_shape(det.poly, shape)   # keep only the in-region part
            if clipped is None:
                continue                               # detection lies fully outside the ROI
            det.poly = clipped                         # region boundary is deliberate: no _refine
            det.length_px, det.area_px = measure_poly(det.poly)
            crop.detections.append(det)
        unit.crops.append(crop)

    judge_unit(unit, settings)
    unit.overlay_scale = min(1.0, OVERLAY_MAX_W / raw_w)
    unit.overlay_base = cv2.resize(raw, None, fx=unit.overlay_scale, fy=unit.overlay_scale)
    unit.crop_bases = [raw[c.window[1]:c.window[3], c.window[0]:c.window[2]].copy()
                       for c in unit.crops]
    return unit, product


# --------------------------------------------------------- template drawing --

class TemplatePane(ttk.Frame):
    """Pan/zoom reference view with Draw and Select tools; boxes saved under a
    product name (versioned)."""

    def __init__(self, master):
        super().__init__(master, padding=4)
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
        self.cmb_product["values"] = products()

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
        self.var_product.set(suggest_product(path))    # editable suggestion
        self._refresh_versions(self.var_product.get())
        self.boxes = list(boxes_for(self.var_product.get(), self.version))
        self._poly = []
        self.redraw()
        return True

    def open_raw(self):
        path = filedialog.askopenfilename(
            title="Reference RAW",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")])
        if path:
            self.load_image(path)

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

    def delete_selected(self, _evt=None):
        """Delete key: remove the picked polygon vertex if one is selected (and the
        polygon keeps >= 3 points); otherwise delete the whole selected shape."""
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
        if len(self._poly) >= 3:
            sh = self.clamp_shape({"type": "poly", "pts": [list(p) for p in self._poly]})
            self.boxes.append(sh)
            self.selected = len(self.boxes) - 1
        self._poly = []
        self.redraw()

    def _cancel_poly(self):
        if self._poly:
            self._poly = []
            self.redraw()

    def _undo_poly_point(self):
        """Drop the last-placed polygon vertex (step back a wrong click)."""
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

VERDICT_TAG = {"NG": "#ffb3b3", "Review": "#f0b3f0",
               "Cleaning": "#ffe9a8", "Overkill": "#b8e6b8"}
ACTIVE_VER = "(active)"


class ScanPane(ttk.Frame):
    """Pick one product + version, scan a folder, view like rejudge_gui."""

    def __init__(self, master, app):
        super().__init__(master, padding=4)
        self.app = app
        self.project = app.project
        self.settings = app.settings
        self.weights = find_weights(self.settings.weights_dirs, self.project)
        self.q = queue.Queue()
        self.worker = None
        self.stop_evt = threading.Event()
        self.units = []
        self.current = None
        self.current_crop = 0
        self._photo = {}

        # controls ---------------------------------------------------------
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="RAW folder…", command=self.pick_raw).pack(side="left")
        self.var_raw = tk.StringVar()
        ttk.Entry(bar, textvariable=self.var_raw, width=30).pack(side="left", padx=4)
        ttk.Label(bar, text="product:").pack(side="left")
        self.var_product = tk.StringVar()
        self.cmb_product = ttk.Combobox(bar, textvariable=self.var_product, width=18,
                                        state="readonly")
        self.cmb_product.pack(side="left", padx=4)
        self.cmb_product.configure(postcommand=self._refresh_products)
        self.cmb_product.bind("<<ComboboxSelected>>", lambda e: self._refresh_versions())
        ttk.Label(bar, text="version:").pack(side="left")
        self.var_version = tk.StringVar(value=ACTIVE_VER)
        self.cmb_version = ttk.Combobox(bar, textvariable=self.var_version, width=10,
                                        state="readonly")
        self.cmb_version.pack(side="left", padx=4)
        self.cmb_version.configure(postcommand=self._refresh_versions)
        ttk.Label(bar, text="weight:").pack(side="left")
        self.var_weight = tk.StringVar()
        self.cmb = ttk.Combobox(bar, textvariable=self.var_weight,
                                values=list(self.weights), width=20, state="readonly")
        self.cmb.pack(side="left", padx=4)
        if self.weights:
            self.cmb.current(0)
        ttk.Label(bar, text="conf:").pack(side="left")
        self.var_conf = tk.StringVar(value=str(self.settings.conf_threshold))
        ttk.Entry(bar, textvariable=self.var_conf, width=5).pack(side="left")
        self.btn = ttk.Button(bar, text="Scan folder", command=self.start)
        self.btn.pack(side="left", padx=8)
        self.var_annot = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Annotations", variable=self.var_annot,
                        command=self._redraw).pack(side="left")
        ttk.Button(bar, text="Open output", command=self.open_output).pack(side="left", padx=6)
        self._refresh_products()
        self.var_status = tk.StringVar(value="Pick a RAW folder, a product and a weight, then Scan.")
        ttk.Label(self, textvariable=self.var_status).pack(fill="x")

        # viewer -----------------------------------------------------------
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, pady=4)
        left = ttk.Frame(pane)
        pane.add(left, weight=1)
        self.tree_units = ttk.Treeview(left, columns=("sv", "sug"), height=22)
        self.tree_units.heading("#0", text="Serial")
        self.tree_units.column("#0", width=120)
        for c, t, w in (("sv", "Station-View", 90), ("sug", "Suggested", 80)):
            self.tree_units.heading(c, text=t)
            self.tree_units.column(c, width=w, anchor="center")
        self.tree_units.pack(fill="both", expand=True)
        self.tree_units.bind("<<TreeviewSelect>>", lambda e: self._select_unit())
        for v, col in VERDICT_TAG.items():
            self.tree_units.tag_configure(v, background=col)

        right = ttk.Frame(pane)
        pane.add(right, weight=4)
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)
        self.cv_overlay = tk.Canvas(self.nb, bg="#222", highlightthickness=0)
        self.nb.add(self.cv_overlay, text="Full-image overlay")
        crop_tab = ttk.Frame(self.nb)
        self.nb.add(crop_tab, text="Crops")
        self.lst_crops = tk.Listbox(crop_tab, width=26)
        self.lst_crops.pack(side="left", fill="y")
        self.lst_crops.bind("<<ListboxSelect>>", lambda e: self._select_crop())
        self.cv_crop = tk.Canvas(crop_tab, bg="#222", highlightthickness=0)
        self.cv_crop.pack(side="left", fill="both", expand=True)
        self.cv_overlay.bind("<Configure>", lambda e: self._redraw())
        self.cv_crop.bind("<Configure>", lambda e: self._redraw())

        det_cols = ("crop", "cls", "conf", "measure", "verdict", "note")
        self.tree_dets = ttk.Treeview(right, columns=det_cols, show="headings", height=6)
        for c, t, w in (("crop", "Box", 45), ("cls", "Class", 100), ("conf", "Conf", 55),
                        ("measure", "Size", 100), ("verdict", "Verdict", 80), ("note", "Note", 460)):
            self.tree_dets.heading(c, text=t)
            self.tree_dets.column(c, width=w, anchor="w")
        self.tree_dets.pack(fill="x")

        self.after(120, self._drain)

    def refresh_all(self):
        self._refresh_products()
        self._refresh_versions()

    def _refresh_products(self):
        vals = products()
        self.cmb_product["values"] = vals
        if not self.var_product.get() and vals:
            self.var_product.set(vals[0])

    def _refresh_versions(self):
        prod = self.var_product.get()
        vals = [ACTIVE_VER] + (versions_of(prod) if prod else [])
        self.cmb_version["values"] = vals
        if self.var_version.get() not in vals:
            self.var_version.set(ACTIVE_VER)

    def pick_raw(self):
        d = filedialog.askdirectory(title="RAW image folder")
        if d:
            self.var_raw.set(d)

    def open_output(self):
        out = os.path.join(self.project, self.settings.output_dir)
        os.makedirs(out, exist_ok=True)
        os.startfile(out)

    # ---- worker -> queue --------------------------------------------------
    def start(self):
        if self.worker and self.worker.is_alive():
            self.stop_evt.set()
            self.btn["state"] = "disabled"
            self.var_status.set("Stopping…")
            return
        raw_dir = self.var_raw.get().strip()
        if not os.path.isdir(raw_dir):
            messagebox.showerror("No folder", "Pick a RAW image folder.")
            return
        product = self.var_product.get().strip()
        if not product:
            messagebox.showerror("No product", "Pick a product (draw one in the Template tab).")
            return
        if not self.var_weight.get():
            messagebox.showerror("No weight", "Pick a trained weight (.pt).")
            return
        try:
            self.settings.conf_threshold = float(self.var_conf.get())
        except ValueError:
            messagebox.showerror("Bad conf", "conf must be a number.")
            return
        chosen = self.var_version.get()
        version = None if chosen == ACTIVE_VER else chosen
        if not boxes_for(product, version):
            if messagebox.askyesno("No master box",
                                   f"Product '{product}' has no boxes for {chosen}.\n\n"
                                   f"Draw it now in the Template tab?"):
                self.app.draw_missing(product, None)
            return
        self.units.clear()
        self.current, self.current_crop = None, 0
        self.tree_units.delete(*self.tree_units.get_children())
        self.tree_dets.delete(*self.tree_dets.get_children())
        self.lst_crops.delete(0, "end")
        self.cv_overlay.delete("all")
        self.cv_crop.delete("all")
        self.btn["text"] = "Stop"
        self.var_status.set("Loading model…")
        engine = Engine(self.weights[self.var_weight.get()], self.settings)
        writer = DateWriter(os.path.join(self.project, self.settings.output_dir))
        self.stop_evt = threading.Event()
        self.worker = threading.Thread(
            target=self._scan, args=(engine, raw_dir, writer, product, version, self.stop_evt),
            daemon=True)
        self.worker.start()

    def _scan(self, engine, raw_dir, writer, product, version, stop_evt):
        try:
            engine.load()
        except Exception as e:                         # noqa: BLE001
            self.q.put(("status", f"FATAL: could not load weight: {e}"))
            self.q.put(("done",))
            return
        raws = sorted(glob.glob(os.path.join(raw_dir, f"*{IMAGE_EXT}")))
        vtag = version or ACTIVE_VER
        self.q.put(("status", f"{len(raws)} RAW file(s), product {product}/{vtag} — scanning…"))
        done = 0
        for path in raws:
            if stop_evt.is_set():
                break
            try:
                unit, _ = process_raw(engine, path, self.settings, product, version)
            except Exception as e:                     # noqa: BLE001
                self.q.put(("status", f"ERROR {os.path.basename(path)}: {e}"))
                continue
            if unit is None:                           # product emptied mid-scan
                continue
            writer.save_images(unit)
            writer.write_all([unit], only_date=unit.date)
            done += 1
            self.q.put(("unit", unit))
        self.q.put(("status", f"Done. scanned {done} with {product}/{vtag}."))
        self.q.put(("done",))

    def _drain(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "unit":
                    self._add_unit(msg[1])
                elif msg[0] == "status":
                    self.var_status.set(msg[1])
                elif msg[0] == "done":
                    self.btn["state"] = "normal"
                    self.btn["text"] = "Scan folder"
        except queue.Empty:
            pass
        self.after(120, self._drain)

    # ---- viewer -----------------------------------------------------------
    def _add_unit(self, unit):
        idx = len(self.units)
        self.units.append(unit)
        tag = unit.suggested if unit.suggested in VERDICT_TAG else "Review"
        self.tree_units.insert("", "end", iid=str(idx), text=unit.serial,
                               values=(unit.station_view, unit.suggested), tags=(tag,))
        self.tree_units.selection_set(str(idx))
        self.tree_units.see(str(idx))
        self._show_unit(idx)

    def _select_unit(self):
        sel = self.tree_units.selection()
        if sel:
            self._show_unit(int(sel[0]))

    def _show_unit(self, idx):
        if not (0 <= idx < len(self.units)):
            return
        self.current, self.current_crop = idx, 0
        u = self.units[idx]
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
            self.lst_crops.insert("end", f"box {c.index} -> {c.verdict}")
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
        cw = max(50, canvas.winfo_width())
        ch = max(50, canvas.winfo_height())
        h, w = bgr.shape[:2]
        s = min(cw / w, ch / h)
        img = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self._photo[key] = photo
        canvas.delete("all")
        canvas.create_image(cw / 2, ch / 2, image=photo, anchor="center")


# --------------------------------------------------------------------- app --

class App:
    def __init__(self, root):
        self.root = root
        root.title("Surface scan (ROI) - prototype")
        root.geometry("1500x900")
        self.project = app_dir()
        self.settings = Settings.load()

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True)
        self.template = TemplatePane(self.nb)
        self.scan = ScanPane(self.nb, self)
        self.nb.add(self.template, text="Template")
        self.nb.add(self.scan, text="Scan")
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab)

    def _on_tab(self, _evt):
        if self.nb.select() == str(self.scan):
            self.scan.refresh_all()

    def draw_missing(self, product, raw_path):
        if raw_path:
            self.template.load_for(product, raw_path)
        else:
            self.template.var_product.set(product)
        self.nb.select(self.template)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
