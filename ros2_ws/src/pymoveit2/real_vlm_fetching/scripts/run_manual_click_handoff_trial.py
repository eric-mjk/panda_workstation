#!/usr/bin/env python3
"""
Capture → Tkinter click → handoff VLM trial.
No OpenCV GUI. No robot commands. No external API calls by default.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parents[1]   # scripts → real_vlm_fetching → repo root

_DEFAULT_TASK_TEXT = (
    "fetch the target object at the clicked grasp/extraction marker"
    " while disturbing surrounding objects as little as possible"
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _save_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def _run(cmd, cwd=None):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def _click_target_tkinter(image_path: str) -> tuple[int, int] | None:
    """
    Open image in a Tkinter window, let user click the target pixel.
    Returns (u, v) in original image resolution, or None if cancelled.
    Raises RuntimeError if Tkinter or PIL is unavailable.
    """
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError as exc:
        raise RuntimeError(f"Tkinter or PIL not available: {exc}")

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    root = tk.Tk()
    root.title("Click target pixel  |  Enter/s: accept    q/Esc: cancel")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    max_w, max_h = max(640, sw - 80), max(480, sh - 120)
    scale = min(max_w / orig_w, max_h / orig_h, 1.0)
    dw, dh = int(orig_w * scale), int(orig_h * scale)

    disp = img.resize((dw, dh), Image.LANCZOS)

    state = {"u": None, "v": None, "accepted": False, "ids": []}

    canvas = tk.Canvas(root, width=dw, height=dh, cursor="crosshair",
                       highlightthickness=0)
    canvas.pack()
    _tk_img = ImageTk.PhotoImage(disp)
    canvas.create_image(0, 0, anchor="nw", image=_tk_img)
    canvas._ref = _tk_img  # prevent GC

    def on_click(event):
        for cid in state["ids"]:
            canvas.delete(cid)
        state["ids"].clear()
        state["u"] = int(round(event.x / scale))
        state["v"] = int(round(event.y / scale))
        r = 8
        state["ids"] += [
            canvas.create_oval(event.x - r, event.y - r,
                               event.x + r, event.y + r,
                               outline="#ff2020", width=2),
            canvas.create_line(event.x - r * 2, event.y,
                               event.x + r * 2, event.y,
                               fill="#ff2020", width=2),
            canvas.create_line(event.x, event.y - r * 2,
                               event.x, event.y + r * 2,
                               fill="#ff2020", width=2),
        ]
        print(f"  Clicked: u={state['u']}, v={state['v']}")

    def accept(event=None):
        state["accepted"] = True
        root.quit()

    def cancel(event=None):
        root.quit()

    canvas.bind("<Button-1>", on_click)
    root.bind("<Return>", accept)
    root.bind("s", accept)
    root.bind("q", cancel)
    root.bind("<Escape>", cancel)

    root.mainloop()
    try:
        root.destroy()
    except Exception:
        pass

    if state["accepted"] and state["u"] is not None:
        return state["u"], state["v"]
    return None


def _parse_best_clock(handoff_out_dir: Path):
    """Return best_clock from result.json (preferred) or summary.csv, or None."""
    for rj in sorted(handoff_out_dir.rglob("result.json")):
        try:
            data = json.loads(rj.read_text())
            for field in ("best_clock", "selected_clock_direction"):
                val = data.get(field)
                if val not in (None, ""):
                    return int(val)
        except Exception:
            pass

    summary = handoff_out_dir / "summary.csv"
    if summary.exists():
        try:
            with summary.open(newline="") as f:
                for row in csv.DictReader(f):
                    for field in ("best_clock", "selected_clock_direction"):
                        val = (row.get(field) or "").strip()
                        if val:
                            return int(val)
        except Exception:
            pass
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Capture → manual click → handoff VLM trial (no robot commands)."
    )
    p.add_argument("--out-dir", required=True, metavar="PATH")
    p.add_argument("--image", default=None, metavar="PATH",
                   help="Use this image; skip capture.")
    p.add_argument("--capture-if-no-image", action="store_true", default=True,
                   help="Capture from camera when --image is omitted (default: True).")
    p.add_argument("--target-tag", required=True, metavar="TAG")
    p.add_argument("--object-class", required=True, metavar="TEXT")
    p.add_argument("--task-text", default=_DEFAULT_TASK_TEXT, metavar="TEXT")
    p.add_argument("--target-u", type=int, default=None, metavar="INT",
                   help="Skip GUI; use this pixel column.")
    p.add_argument("--target-v", type=int, default=None, metavar="INT",
                   help="Skip GUI; use this pixel row.")
    p.add_argument("--handoff-root",
                   default=str(_REPO_ROOT / "vlm_single_view_fetching"),
                   metavar="PATH")
    p.add_argument("--run-handoff", action="store_true",
                   help="Run the handoff VLM batch runner after clicking.")
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="Pass --dry-run to handoff runner (default: True).")
    p.add_argument("--no-dry-run", action="store_true",
                   help="Run real VLM instead of dry-run.")
    p.add_argument("--camera-serial", default=None, metavar="SERIAL")
    p.add_argument("--warmup-frames", type=int, default=10, metavar="INT")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    handoff_root = Path(args.handoff_root).resolve()

    # 1. Image ─────────────────────────────────────────────────────────────────
    if args.image is not None:
        image_path = Path(args.image).resolve()
    else:
        capture_dir = out_dir / "capture"
        cmd = [
            sys.executable, str(_SCRIPTS_DIR / "capture_once.py"),
            "--out-dir", str(capture_dir),
            "--index", "0",
            "--warmup-frames", str(args.warmup_frames),
        ]
        if args.camera_serial:
            cmd += ["--camera-serial", args.camera_serial]
        _run(cmd)
        image_path = (capture_dir / "rgb" / "00000.png").resolve()

    # 2. Target u/v ────────────────────────────────────────────────────────────
    if args.target_u is not None and args.target_v is not None:
        target_u, target_v = args.target_u, args.target_v
        print(f"Using manual coordinates: u={target_u}, v={target_v}")
    else:
        print(f"Opening image for click selection: {image_path}")
        try:
            result = _click_target_tkinter(str(image_path))
        except Exception as exc:
            print(f"ERROR: click UI failed — {exc}", file=sys.stderr)
            print("Re-run with --target-u and --target-v to skip the GUI.",
                  file=sys.stderr)
            sys.exit(1)
        if result is None:
            print("Cancelled by user.", file=sys.stderr)
            sys.exit(1)
        target_u, target_v = result
        print(f"Selected: u={target_u}, v={target_v}")

    # 3. clicked_target.json ───────────────────────────────────────────────────
    clicked_target_path = out_dir / "clicked_target.json"
    _save_json(clicked_target_path, {
        "image_path": str(image_path),
        "target_provider": "manual_click",
        "tag": args.target_tag,
        "object_class": args.object_class,
        "u": target_u,
        "v": target_v,
    })
    print(f"Wrote: {clicked_target_path}")

    # 4. handoff_target_set.json ───────────────────────────────────────────────
    handoff_target_set_path = out_dir / "handoff_target_set.json"
    _save_json(handoff_target_set_path, {
        "image_path": str(image_path),
        "target_provider": "manual_click",
        "targets": [
            {
                "id": 1,
                "tag": args.target_tag,
                "u": target_u,
                "v": target_v,
                "object_class": args.object_class,
                "task_text": args.task_text,
                "target_provider": "manual_click",
                "target_note": "manual clicked target marker under home/BEV capture pose",
            }
        ],
    })
    print(f"Wrote: {handoff_target_set_path}")

    # 5. Run handoff ───────────────────────────────────────────────────────────
    handoff_out_dir = out_dir / "handoff_output"
    handoff_ran = False
    if args.run_handoff:
        use_dry_run = not args.no_dry_run
        cmd = [
            sys.executable,
            "fetching_baseline/run_batch_vlm_fetching_direction.py",
            "--target-set-json", str(handoff_target_set_path),
            "--out-root", str(handoff_out_dir),
        ]
        if use_dry_run:
            cmd.append("--dry-run")
        _run(cmd, cwd=str(handoff_root))
        handoff_ran = True

    # 6. adapter_manifest.json ─────────────────────────────────────────────────
    use_dry_run = not args.no_dry_run
    _save_json(out_dir / "adapter_manifest.json", {
        "image_path": str(image_path),
        "clicked_target_json": str(clicked_target_path),
        "handoff_target_set_json": str(handoff_target_set_path),
        "handoff_output_dir": str(handoff_out_dir),
        "target_u": target_u,
        "target_v": target_v,
        "target_tag": args.target_tag,
        "object_class": args.object_class,
        "handoff_ran": handoff_ran,
        "dry_run": use_dry_run,
    })

    # 7. Print summary ─────────────────────────────────────────────────────────
    print(f"\nImage                  : {image_path}")
    print(f"Clicked u/v            : u={target_u}, v={target_v}")
    print(f"clicked_target.json    : {clicked_target_path}")
    print(f"handoff_target_set.json: {handoff_target_set_path}")
    print(f"adapter_manifest.json  : {out_dir / 'adapter_manifest.json'}")

    if handoff_ran:
        print(f"handoff_output         : {handoff_out_dir}")
        summary_csv = handoff_out_dir / "summary.csv"
        if summary_csv.exists():
            print(f"summary.csv            : {summary_csv}")
        for rj in sorted(handoff_out_dir.rglob("result.json")):
            print(f"result.json            : {rj}")
        best_clock = _parse_best_clock(handoff_out_dir)
        if best_clock is not None:
            print(f"\n*** best_clock = {best_clock} ***")
        else:
            print("\n(best_clock not available)")


if __name__ == "__main__":
    main()
