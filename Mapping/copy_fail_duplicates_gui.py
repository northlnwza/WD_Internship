"""
GUI wrapper for copy_fail_duplicates logic.

Workflow:
  1. Select root folder and station name.
  2. Click Preview – shows what WOULD be copied (dry-run, no changes).
  3. Review the list, then click Run to actually copy the files.
"""

import re
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


# ──────────────────────────────────────────────────────────────────────────────
# Core logic (same as CLI script, returns results instead of printing)
# ──────────────────────────────────────────────────────────────────────────────

def sn_of(path: Path) -> str:
    return path.name.split("_", 1)[0]


def collect_work(root: Path, station: str):
    """
    Return a list of (src_path, dest_path) pairs that need to be copied,
    plus a list of summary strings per day.
    Raises ValueError with a message if the folder layout is wrong.
    """
    fail_station = root / "Fail" / station
    image_station = root / "Image" / station

    if not fail_station.is_dir():
        raise ValueError(f"Fail station folder not found:\n{fail_station}")
    if not image_station.is_dir():
        raise ValueError(f"Image station folder not found:\n{image_station}")

    pairs = []          # (src, dest)
    day_summaries = []  # human-readable lines
    day_pattern = re.compile(r"^\d{8}$")

    for day_dir in sorted(p for p in fail_station.iterdir() if p.is_dir()):
        if not day_pattern.match(day_dir.name):
            continue

        fail_sns = {sn_of(f) for f in day_dir.rglob("*.jpg")}
        if not fail_sns:
            day_summaries.append(f"  {day_dir.name}: 0 Fail SNs – skipped")
            continue

        img_day = image_station / day_dir.name
        if not img_day.is_dir():
            day_summaries.append(
                f"  {day_dir.name}: {len(fail_sns)} Fail SNs, no Image folder – skipped"
            )
            continue

        img_pics = [
            f for f in img_day.rglob("*.jpg")
            if "fail_copy" not in f.parts
        ]
        matches = [f for f in img_pics if sn_of(f) in fail_sns]
        dest_root = day_dir / "failed_raw"

        for f in matches:
            pairs.append((f, dest_root / f.name))

        day_summaries.append(
            f"  {day_dir.name}: {len(fail_sns)} Fail SNs, "
            f"{len(img_pics)} Image pictures → {len(matches)} to copy"
        )

    return pairs, day_summaries


def do_copy(pairs):
    """Execute the file copies. Returns count of files copied."""
    for src, dest in pairs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        shutil.copy2(src, dest)
    return len(pairs)


# ──────────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Copy Fail Duplicates")
        self.resizable(True, True)
        self.minsize(640, 480)

        self._pending_pairs = []   # set by Preview, consumed by Run
        self._build_ui()
        self._set_status("Ready – select a root folder and click Preview.")

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # ── top frame: inputs ─────────────────────────────────────────────
        top = ttk.LabelFrame(self, text="Settings")
        top.pack(fill=tk.X, **pad)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Root folder:").grid(row=0, column=0, sticky=tk.W, **pad)
        self._root_var = tk.StringVar()
        ttk.Entry(top, textvariable=self._root_var).grid(
            row=0, column=1, sticky=tk.EW, **pad
        )
        ttk.Button(top, text="Browse…", command=self._browse_root).grid(
            row=0, column=2, **pad
        )

        ttk.Label(top, text="Station name:").grid(row=1, column=0, sticky=tk.W, **pad)
        self._station_var = tk.StringVar(value="ST5_SEW-TOP")
        ttk.Entry(top, textvariable=self._station_var).grid(
            row=1, column=1, sticky=tk.EW, **pad
        )

        # ── middle frame: results text ────────────────────────────────────
        mid = ttk.LabelFrame(self, text="Preview / Results")
        mid.pack(fill=tk.BOTH, expand=True, **pad)

        self._text = tk.Text(
            mid, wrap=tk.NONE, state=tk.DISABLED,
            font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white"
        )
        vsb = ttk.Scrollbar(mid, orient=tk.VERTICAL, command=self._text.yview)
        hsb = ttk.Scrollbar(mid, orient=tk.HORIZONTAL, command=self._text.xview)
        self._text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self._text.pack(fill=tk.BOTH, expand=True)

        # colour tags
        self._text.tag_config("header",  foreground="#9cdcfe")
        self._text.tag_config("day",     foreground="#ce9178")
        self._text.tag_config("file",    foreground="#b5cea8")
        self._text.tag_config("total",   foreground="#dcdcaa")
        self._text.tag_config("error",   foreground="#f44747")
        self._text.tag_config("success", foreground="#4ec9b0")

        # ── bottom bar: buttons + status ──────────────────────────────────
        bot = ttk.Frame(self)
        bot.pack(fill=tk.X, **pad)

        self._preview_btn = ttk.Button(
            bot, text="Preview", command=self._on_preview
        )
        self._preview_btn.pack(side=tk.LEFT, padx=(0, 4))

        self._run_btn = ttk.Button(
            bot, text="Run (copy files)", command=self._on_run, state=tk.DISABLED
        )
        self._run_btn.pack(side=tk.LEFT, padx=4)

        ttk.Button(bot, text="Close", command=self.destroy).pack(side=tk.RIGHT)

        self._status_var = tk.StringVar()
        ttk.Label(bot, textvariable=self._status_var, anchor=tk.W).pack(
            side=tk.LEFT, padx=12, fill=tk.X, expand=True
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _browse_root(self):
        folder = filedialog.askdirectory(title="Select root folder (contains Fail\\ and Image\\)")
        if folder:
            self._root_var.set(folder)
            self._run_btn.config(state=tk.DISABLED)
            self._pending_pairs = []

    def _set_status(self, msg: str):
        self._status_var.set(msg)

    def _append(self, text: str, tag: str = ""):
        self._text.config(state=tk.NORMAL)
        if tag:
            self._text.insert(tk.END, text, tag)
        else:
            self._text.insert(tk.END, text)
        self._text.see(tk.END)
        self._text.config(state=tk.DISABLED)

    def _clear_text(self):
        self._text.config(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.config(state=tk.DISABLED)

    # ── Preview ───────────────────────────────────────────────────────────────

    def _on_preview(self):
        root_str = self._root_var.get().strip()
        station = self._station_var.get().strip()

        if not root_str:
            messagebox.showwarning("Missing input", "Please select a root folder first.")
            return
        if not station:
            messagebox.showwarning("Missing input", "Please enter a station name.")
            return

        self._pending_pairs = []
        self._run_btn.config(state=tk.DISABLED)
        self._preview_btn.config(state=tk.DISABLED)
        self._set_status("Scanning…")
        self._clear_text()

        threading.Thread(
            target=self._run_preview,
            args=(Path(root_str), station),
            daemon=True
        ).start()

    def _run_preview(self, root: Path, station: str):
        try:
            pairs, day_summaries = collect_work(root, station)
        except ValueError as exc:
            self.after(0, self._preview_error, str(exc))
            return

        self.after(0, self._preview_done, pairs, day_summaries, root, station)

    def _preview_error(self, msg: str):
        self._append("ERROR\n\n", "error")
        self._append(msg + "\n", "error")
        self._set_status("Error – check the folder paths above.")
        self._preview_btn.config(state=tk.NORMAL)

    def _preview_done(self, pairs, day_summaries, root: Path, station: str):
        self._append(f"Root:    {root}\n", "header")
        self._append(f"Station: {station}\n\n", "header")

        self._append("── Day summary ──────────────────────────────────────\n", "header")
        for line in day_summaries:
            self._append(line + "\n", "day")

        self._append(
            f"\n── Files that WOULD be copied ({len(pairs)} total) ──────────\n",
            "header"
        )
        if pairs:
            for src, dest in pairs:
                self._append(f"  {src.name}\n", "file")
                self._append(f"    → {dest}\n")
        else:
            self._append("  (nothing to copy)\n")

        self._append(f"\nTotal: {len(pairs)} file(s) would be copied.\n", "total")
        self._append(
            "\nReview the list above, then click  Run  to copy the files,\n"
            "or close the window to cancel.\n",
            "total"
        )

        self._pending_pairs = pairs
        self._preview_btn.config(state=tk.NORMAL)

        if pairs:
            self._run_btn.config(state=tk.NORMAL)
            self._set_status(f"Preview complete – {len(pairs)} file(s) ready to copy. Click Run to proceed.")
        else:
            self._set_status("Preview complete – nothing to copy.")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _on_run(self):
        if not self._pending_pairs:
            messagebox.showinfo("Nothing to do", "Run Preview first to load the file list.")
            return

        answer = messagebox.askyesno(
            "Confirm copy",
            f"Copy {len(self._pending_pairs)} file(s) now?\n\nThis cannot be undone.",
        )
        if not answer:
            return

        self._run_btn.config(state=tk.DISABLED)
        self._preview_btn.config(state=tk.DISABLED)
        self._set_status("Copying…")

        pairs = list(self._pending_pairs)
        threading.Thread(
            target=self._run_copy,
            args=(pairs,),
            daemon=True
        ).start()

    def _run_copy(self, pairs):
        try:
            count = do_copy(pairs)
            self.after(0, self._copy_done, count)
        except Exception as exc:
            self.after(0, self._copy_error, str(exc))

    def _copy_done(self, count: int):
        self._append(f"\n✓ Done – {count} file(s) copied successfully.\n", "success")
        self._set_status(f"Done – {count} file(s) copied.")
        self._pending_pairs = []
        self._preview_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Done", f"{count} file(s) copied successfully.")

    def _copy_error(self, msg: str):
        self._append(f"\nERROR during copy:\n{msg}\n", "error")
        self._set_status("Copy failed – see details above.")
        self._preview_btn.config(state=tk.NORMAL)
        self._run_btn.config(state=tk.NORMAL)


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
