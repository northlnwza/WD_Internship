"""
Model Tester — desktop GUI for the Scratch vs Contam YOLO segmentation pipeline.

This app lives in its own folder and reads data (models + images) from the
sibling ML_pipeline/ project. Pick a trained model (.pt), point it at a folder
of images, run segmentation, then browse results — original vs. annotated.

Data root (where models/ and test_result/ live) is auto-detected:
  1. env var  ML_PIPELINE_DIR  (if set)
  2. ../ML_pipeline   (sibling of this app folder)   ← default layout
  3. ./ML_pipeline
  4. parent folder
You can always Browse… to any model/folder regardless of the guess.

Run:
    python model_tester.py           (or double-click run_app.bat)
"""

import os
import threading
import queue
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

APP_DIR = Path(__file__).parent.resolve()
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def find_data_root(app_dir: Path) -> Path:
    """Locate the ML_pipeline data folder (models/ + test_result/)."""
    candidates = []
    env = os.environ.get("ML_PIPELINE_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [
        app_dir.parent / "ML_pipeline",   # sibling  (default layout)
        app_dir / "ML_pipeline",          # nested
        app_dir.parent,                   # parent
    ]
    for c in candidates:
        try:
            c = c.resolve()
        except OSError:
            continue
        if c.is_dir() and ((c / "models").is_dir()
                           or (c / "test_result").is_dir()
                           or any(c.glob("*.pt"))):
            return c
    return app_dir.parent.resolve()


DATA_ROOT = find_data_root(APP_DIR)
DEFAULT_FOLDER = DATA_ROOT / "test_result"


# ── model / file discovery ──────────────────────────────────────────────────

def discover_models(root: Path):
    """Return candidate .pt weight files under the data root (trained first)."""
    found = []
    found += sorted(root.glob("models/**/weights/*.pt"))   # trained checkpoints
    found += sorted(root.glob("*.pt"))                      # base weights in root
    seen, uniq = set(), []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def label_for(path: Path, root: Path) -> str:
    """A short, human label for a model path (relative to data root when possible)."""
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def list_images(folder: Path):
    imgs = []
    for ext in IMAGE_EXTS:
        imgs += folder.glob(f"*{ext}")
        imgs += folder.glob(f"*{ext.upper()}")
    uniq = {p.resolve(): p for p in imgs}     # de-dup case-insensitive overlaps
    return sorted(uniq.values(), key=lambda p: p.name.lower())


# ── class definitions ────────────────────────────────────────────────────────
# The trained model reports raw class names (order set in ML_pipeline/config.yaml).
# Map each raw name → a readable label + defect family, so the UI shows friendly
# names and can group by family. Any class the model reports but isn't listed
# here falls back to its raw name (see class_label / class_group).
CLASS_DEFS = {
    "comtam_dust":  ("Contam · Dust",  "Contamination"),
    "comtam_stain": ("Contam · Stain", "Contamination"),
    "comtam_x":     ("Contam · Misc",  "Contamination"),
    "scratch_dent": ("Scratch · Dent", "Scratch"),
    "scratch_line": ("Scratch · Line", "Scratch"),
    "other":        ("Other",          "Other"),
}


def class_label(raw: str) -> str:
    """Readable label for a raw model class name (raw name if undefined)."""
    d = CLASS_DEFS.get(raw)
    return d[0] if d else raw


def class_group(raw: str) -> str:
    """Defect family for a raw model class name ('Other' if undefined)."""
    d = CLASS_DEFS.get(raw)
    return d[1] if d else "Other"


# ── the app ─────────────────────────────────────────────────────────────────

class ModelTester(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Model Tester — Scratch vs Contam")
        self.geometry("1180x760")
        self.minsize(940, 600)

        # runtime state
        self.model_paths = {}          # label -> Path
        self.model_cache = {}          # resolved-path-str -> loaded YOLO model
        self.results = {}              # img-path-str -> {original, annotated, detections, error}
        self.image_order = []          # list[Path], matches listbox order
        self.current_key = None        # currently displayed img-path-str
        self.view_mode = tk.StringVar(value="result")
        self.conf_var = tk.DoubleVar(value=0.25)
        self.folder_var = tk.StringVar(value=str(DEFAULT_FOLDER))
        self.status_var = tk.StringVar(value=f"Data root: {DATA_ROOT}")
        self.queue = queue.Queue()
        self.worker = None
        self.cancel_event = threading.Event()
        self._display_imgtk = None     # keep a ref so Tk doesn't GC the image
        self._resize_job = None

        self._build_ui()
        self._refresh_models(select_trained=True)
        self.after(100, self._drain_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- UI construction ----------------------------------------------------

    def _build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use("vista")   # native-looking on Windows
        except tk.TclError:
            pass
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        # top control bar
        top = ttk.Frame(self, padding=(10, 10, 10, 6))
        top.pack(side="top", fill="x")

        # row 0 — model
        ttk.Label(top, text="Model:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self.model_combo = ttk.Combobox(top, state="readonly", width=42)
        self.model_combo.grid(row=0, column=1, sticky="we", pady=3)
        ttk.Button(top, text="Browse…", command=self._browse_model).grid(row=0, column=2, padx=6)
        ttk.Button(top, text="↻", width=3, command=lambda: self._refresh_models()).grid(row=0, column=3)

        ttk.Label(top, text="Confidence:").grid(row=0, column=4, sticky="e", padx=(18, 6))
        self.conf_scale = ttk.Scale(top, from_=0.05, to=0.95, variable=self.conf_var,
                                    command=self._on_conf, length=140)
        self.conf_scale.grid(row=0, column=5, sticky="w")
        self.conf_label = ttk.Label(top, text="0.25", width=5)
        self.conf_label.grid(row=0, column=6, sticky="w")

        # row 1 — folder
        ttk.Label(top, text="Images:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        self.folder_entry = ttk.Entry(top, textvariable=self.folder_var)
        self.folder_entry.grid(row=1, column=1, columnspan=3, sticky="we", pady=3)
        ttk.Button(top, text="Browse…", command=self._browse_folder).grid(row=1, column=4, padx=6)
        self.run_btn = ttk.Button(top, text="▶  Run", style="Accent.TButton", command=self._on_run)
        self.run_btn.grid(row=1, column=5, columnspan=2, sticky="we", padx=(6, 0))

        top.columnconfigure(1, weight=1)

        # main split — image list  |  preview
        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(side="top", fill="both", expand=True, padx=10, pady=(4, 6))

        # left: image list
        left = ttk.Frame(body)
        ttk.Label(left, text="Images", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        list_wrap = ttk.Frame(left)
        list_wrap.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(list_wrap, activestyle="dotbox", exportselection=False,
                                  font=("Consolas", 9))
        sb = ttk.Scrollbar(list_wrap, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        body.add(left, weight=1)

        # right: preview + detections
        right = ttk.Frame(body)

        header = ttk.Frame(right)
        header.pack(fill="x")
        self.title_label = ttk.Label(header, text="No image selected",
                                     font=("Segoe UI", 10, "bold"))
        self.title_label.pack(side="left")
        ttk.Radiobutton(header, text="Result", value="result",
                        variable=self.view_mode, command=self._redisplay).pack(side="right")
        ttk.Radiobutton(header, text="Original", value="original",
                        variable=self.view_mode, command=self._redisplay).pack(side="right", padx=(0, 8))

        self.canvas = tk.Canvas(right, background="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=(4, 4))
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        det_wrap = ttk.LabelFrame(right, text="Detections", padding=6)
        det_wrap.pack(fill="x")
        self.det_text = tk.Text(det_wrap, height=5, font=("Consolas", 9),
                                state="disabled", wrap="none")
        self.det_text.pack(fill="x")

        body.add(right, weight=3)

        # bottom status bar
        bottom = ttk.Frame(self, padding=(10, 4))
        bottom.pack(side="bottom", fill="x")
        self.progress = ttk.Progressbar(bottom, mode="determinate", length=200)
        self.progress.pack(side="right")
        self.save_btn = ttk.Button(bottom, text="Save results…", command=self._save_results,
                                   state="disabled")
        self.save_btn.pack(side="right", padx=8)
        ttk.Button(bottom, text="Classes…", command=self._show_classes).pack(side="left", padx=(0, 8))
        ttk.Label(bottom, textvariable=self.status_var, anchor="w").pack(side="left", fill="x")

    # ---- model handling -----------------------------------------------------

    def _refresh_models(self, select_trained=False):
        paths = discover_models(DATA_ROOT)
        self.model_paths = {label_for(p, DATA_ROOT): p for p in paths}
        labels = list(self.model_paths.keys())
        self.model_combo["values"] = labels
        if not labels:
            self.status_var.set(f"No .pt models under {DATA_ROOT}. Browse… to a model file.")
            return
        pick = 0
        if select_trained:
            for i, lbl in enumerate(labels):
                if lbl.endswith("best.pt"):
                    pick = i
                    break
        self.model_combo.current(pick)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Choose a model (.pt)",
            initialdir=str(DATA_ROOT),
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        lbl = label_for(p, DATA_ROOT)
        self.model_paths[lbl] = p
        self.model_combo["values"] = list(self.model_paths.keys())
        self.model_combo.set(lbl)

    def _selected_model_path(self):
        lbl = self.model_combo.get()
        return self.model_paths.get(lbl)

    # ---- input folder -------------------------------------------------------

    def _browse_folder(self):
        start = self.folder_var.get()
        if not Path(start).is_dir():
            start = str(DATA_ROOT)
        folder = filedialog.askdirectory(title="Choose image folder", initialdir=start)
        if folder:
            self.folder_var.set(folder)

    def _on_conf(self, _=None):
        self.conf_label.config(text=f"{self.conf_var.get():.2f}")

    # ---- run ----------------------------------------------------------------

    def _on_run(self):
        if self.worker and self.worker.is_alive():
            self.cancel_event.set()                 # second click = stop
            self.status_var.set("Stopping after current image…")
            return

        model_path = self._selected_model_path()
        if not model_path or not Path(model_path).exists():
            messagebox.showerror("No model", "Pick a valid .pt model first.")
            return

        folder = Path(self.folder_var.get())
        if not folder.is_dir():
            messagebox.showerror("No folder", f"Not a folder:\n{folder}")
            return

        images = list_images(folder)
        if not images:
            messagebox.showwarning("No images",
                                   f"No images ({', '.join(IMAGE_EXTS)}) in:\n{folder}")
            return

        # reset UI state for a fresh run
        self.results.clear()
        self.image_order = images
        self.current_key = None
        self.listbox.delete(0, "end")
        for img in images:
            self.listbox.insert("end", f"  … {img.name}")
            self.listbox.itemconfig("end", foreground="#888888")
        self._clear_preview()
        self.save_btn.config(state="disabled")
        self.progress.config(maximum=len(images), value=0)
        self.run_btn.config(text="■  Stop")
        self.cancel_event.clear()

        conf = float(self.conf_var.get())
        self.worker = threading.Thread(
            target=self._run_worker,
            args=(Path(model_path), images, conf),
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, model_path: Path, images, conf: float):
        """Background thread: load model, predict per image, push results to queue."""
        try:
            key = str(model_path.resolve())
            model = self.model_cache.get(key)
            if model is None:
                self.queue.put(("status", f"Loading model  {model_path.name} …"))
                from ultralytics import YOLO       # imported here → GUI opens instantly
                model = YOLO(str(model_path))
                self.model_cache[key] = model
            names = model.names

            for i, img_path in enumerate(images, start=1):
                if self.cancel_event.is_set():
                    self.queue.put(("status", "Stopped."))
                    break
                self.queue.put(("status", f"Predicting {i}/{len(images)}  —  {img_path.name}"))
                try:
                    res = model.predict(str(img_path), conf=conf, verbose=False)[0]
                    annotated_bgr = res.plot()                      # numpy BGR
                    annotated = Image.fromarray(annotated_bgr[:, :, ::-1])  # -> RGB
                    original = Image.open(img_path).convert("RGB")
                    dets = []
                    if res.boxes is not None:
                        for c, cf in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                            dets.append((names.get(int(c), str(int(c))), float(cf)))
                    self.queue.put(("result", str(img_path), i - 1,
                                    original, annotated, dets, None))
                except Exception as e:                               # per-image failure
                    self.queue.put(("result", str(img_path), i - 1,
                                    None, None, [], str(e)))
                self.queue.put(("progress", i))
        except Exception:
            self.queue.put(("fatal", traceback.format_exc()))
        finally:
            self.queue.put(("done", None))

    # ---- queue pump (main thread) ------------------------------------------

    def _drain_queue(self):
        try:
            while True:
                kind, *payload = self.queue.get_nowait()
                if kind == "status":
                    self.status_var.set(payload[0])
                elif kind == "progress":
                    self.progress.config(value=payload[0])
                elif kind == "result":
                    self._store_result(*payload)
                elif kind == "fatal":
                    messagebox.showerror("Prediction failed", payload[0])
                elif kind == "done":
                    self._on_worker_done()
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _store_result(self, key, index, original, annotated, dets, error):
        self.results[key] = {
            "original": original, "annotated": annotated,
            "detections": dets, "error": error,
        }
        name = Path(key).name
        if error:
            text, color = f"  ✗ {name}", "#c0392b"
        elif dets:
            text, color = f"  ● {name}  ({len(dets)} det)", "#c0392b"
        else:
            text, color = f"  ○ {name}  (clean)", "#2e7d32"
        self.listbox.delete(index)
        self.listbox.insert(index, text)
        self.listbox.itemconfig(index, foreground=color)
        if self.current_key is None:                # auto-show first finished image
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(index)
            self._display(key)

    def _on_worker_done(self):
        self.run_btn.config(text="▶  Run")
        done = sum(1 for v in self.results.values() if v)
        with_det = sum(1 for v in self.results.values() if v and v["detections"])
        if not self.cancel_event.is_set():
            self.status_var.set(f"Done. {done} image(s) processed, {with_det} with detections.")
        if self.results:
            self.save_btn.config(state="normal")

    # ---- preview display ----------------------------------------------------

    def _on_select(self, _evt):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < len(self.image_order):
            self._display(str(self.image_order[idx]))

    def _redisplay(self):
        if self.current_key:
            self._display(self.current_key)

    def _display(self, key):
        self.current_key = key
        data = self.results.get(key)
        name = Path(key).name
        self.title_label.config(text=name)
        if not data:
            self._clear_preview(keep_title=True)
            self._set_detections("(still processing…)")
            return
        if data["error"]:
            self._clear_preview(keep_title=True)
            self._set_detections(f"ERROR: {data['error']}")
            return

        pil = data["annotated"] if self.view_mode.get() == "result" else data["original"]
        self._render(pil)

        if data["detections"]:
            lines = [f"{class_label(cls):<16} {conf:6.1%}   [{class_group(cls)}]"
                     for cls, conf in sorted(data["detections"], key=lambda d: -d[1])]
            self._set_detections("\n".join(lines))
        else:
            self._set_detections("No defects detected.")

    def _render(self, pil: Image.Image):
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        if cw <= 1 or ch <= 1:            # canvas not laid out yet
            self.after(50, lambda: self._render(pil))
            return
        scale = min(cw / pil.width, ch / pil.height)
        w, h = max(1, int(pil.width * scale)), max(1, int(pil.height * scale))
        resized = pil.resize((w, h), Image.LANCZOS)
        self._display_imgtk = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._display_imgtk, anchor="center")

    def _clear_preview(self, keep_title=False):
        self.canvas.delete("all")
        self._display_imgtk = None
        if not keep_title:
            self.title_label.config(text="No image selected")
            self._set_detections("")

    def _set_detections(self, text):
        self.det_text.config(state="normal")
        self.det_text.delete("1.0", "end")
        self.det_text.insert("1.0", text)
        self.det_text.config(state="disabled")

    def _on_canvas_resize(self, _evt):
        if self._resize_job:                        # debounce resize re-render
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(120, self._redisplay)

    # ---- save ---------------------------------------------------------------

    def _save_results(self):
        annotated = {k: v for k, v in self.results.items()
                     if v and v["annotated"] is not None}
        if not annotated:
            messagebox.showinfo("Nothing to save", "No annotated results yet.")
            return
        default_out = Path(self.folder_var.get()).name + "_output"
        out = filedialog.askdirectory(
            title="Save annotated images to…",
            initialdir=str(Path(self.folder_var.get()).parent / default_out),
        )
        if not out:
            return
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for key, data in annotated.items():
            try:
                data["annotated"].save(out_dir / Path(key).name)
                n += 1
            except Exception as e:
                print(f"save failed for {key}: {e}")
        self.status_var.set(f"Saved {n} annotated image(s) → {out_dir}")
        messagebox.showinfo("Saved", f"Saved {n} annotated image(s) to:\n{out_dir}")

    # ---- class definitions --------------------------------------------------

    def _show_classes(self):
        """Popup listing the defect classes: raw model name → label + family."""
        win = tk.Toplevel(self)
        win.title("Defect classes")
        win.geometry("480x320")
        win.transient(self)
        frm = ttk.Frame(win, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Classes the model can detect",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))

        cols = ("raw", "label", "group")
        heads = {"raw": "model name", "label": "label", "group": "family"}
        tree = ttk.Treeview(frm, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (140, 160, 140)):
            tree.heading(c, text=heads[c])
            tree.column(c, width=w, anchor="w")
        for raw, (label, group) in CLASS_DEFS.items():
            tree.insert("", "end", values=(raw, label, group))
        tree.pack(fill="both", expand=True)

        # flag any class a loaded model reports that the app hasn't defined
        model_names = set()
        for mdl in self.model_cache.values():
            model_names.update(mdl.names.values())
        undefined = sorted(model_names - set(CLASS_DEFS))
        note = ("All model classes are defined." if not undefined else
                "Undefined (shown as raw name): " + ", ".join(undefined))
        ttk.Label(frm, text=note, foreground="#888888").pack(anchor="w", pady=(6, 0))
        ttk.Button(frm, text="Close", command=win.destroy).pack(anchor="e", pady=(8, 0))

    # ---- lifecycle ----------------------------------------------------------

    def _on_close(self):
        self.cancel_event.set()
        self.destroy()


if __name__ == "__main__":
    ModelTester().mainloop()
